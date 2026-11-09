"""Gemini Version: Fit Score ranking, role filtering, location merging."""
import os
import re
import time
import unicodedata
from datetime import date
import pandas as pd
import yaml
from jobspy import scrape_jobs

def clean(text):
    """Lowercase and remove accents."""
    text = unicodedata.normalize("NFKD", str(text or ""))
    return "".join(ch for ch in text if not unicodedata.combining(ch)).lower()

def col(df, name):
    """Return a text column, or empty text if missing."""
    if name in df.columns:
        return df[name].fillna("").astype(str)
    return pd.Series([""] * len(df), index=df.index)

def whole_word(text, keywords):
    """Matches whole word with optional plural 's'."""
    text = clean(text)
    for word in keywords:
        w = re.escape(clean(word))
        if re.search(rf"\b{w}s?\b", text):
            return word
    return ""

def word_start(text, keywords):
    """Matches keyword at start of a word."""
    text = clean(text)
    for word in keywords:
        w = re.escape(clean(word))
        if re.search(rf"\b{w}", text):
            return word
    return ""

def is_priority(company, names):
    company = clean(company)
    for n in names:
        if re.search(rf"\b{re.escape(clean(n))}\b", company):
            return True
    return False

def main():
    with open("criteria.yaml", encoding="utf-8") as f:
        c = yaml.safe_load(f)
        
    fetch_desc = c.get("linkedin_fetch_description", True)
    give_up_after = c.get("skip_site_after_failures", 4)
    sites = list(c.get("sites", []))
    tries = {s: 0 for s in sites}
    wins = {s: 0 for s in sites}
    found = {s: 0 for s in sites}
    skipped = set()
    log = []
    frames = []

    for loc in c["locations"]:
        for term in c["search_terms"]:
            for site in sites:
                if site in skipped:
                    continue
                if site == "zip_recruiter" and loc["country"] not in ("USA", "Canada"):
                    continue
                
                label = f"{site} | {term} | {loc['name']}"
                tries[site] += 1
                
                try:
                    kwargs = dict(
                        site_name=[site],
                        search_term=term,
                        google_search_term=f"{term} jobs near {loc['name']} since yesterday",
                        location=loc["name"],
                        results_wanted=c.get("results_per_search", 20),
                        hours_old=c.get("days_back", 7) * 24,
                        country_indeed=loc["country"],
                    )
                    if site == "linkedin" and fetch_desc:
                        kwargs["linkedin_fetch_description"] = True
                        
                    df = scrape_jobs(**kwargs)
                    
                    if df is None or df.empty:
                        log.append(f"EMPTY  {label}")
                    else:
                        df["searched_location"] = loc["name"]
                        frames.append(df)
                        wins[site] += 1
                        found[site] += len(df)
                        log.append(f"OK     {label} -> {len(df)} jobs")
                except Exception as err:
                    log.append(f"FAILED {label} -> {str(err)[:120]}")
                    
                if wins[site] == 0 and tries[site] >= give_up_after:
                    skipped.add(site)
                    log.append(f"SKIPPED {site}: no jobs after {tries[site]} tries")
                time.sleep(2)

    os.makedirs("results", exist_ok=True)
    log.append("")
    log.append("=== SITE SUMMARY ===")
    for s in sites:
        state = "skipped" if s in skipped else "used"
        log.append(f"{s}: {found[s]} jobs from {wins[s]}/{tries[s]} searches ({state})")

    if not frames:
        log.append("No jobs found from any site.")
        with open("results/run_log.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(log))
        print("\n".join(log))
        return

    jobs = pd.concat(frames, ignore_index=True)
    
    # Merge locations for identical jobs (same title & company)
    jobs['job_id_temp'] = jobs['title'].astype(str) + " | " + jobs['company'].astype(str)
    location_map = jobs.groupby('job_id_temp')['location'].apply(
        lambda x: ' | '.join(sorted(set(x.dropna().astype(str))))
    )
    jobs['location'] = jobs['job_id_temp'].map(location_map)
    jobs = jobs.drop_duplicates(subset=['job_id_temp']).reset_index(drop=True)

    log.append("")
    log.append("=== FILTERS ===")
    log.append(f"Total after merging duplicates: {len(jobs)}")

    title = col(jobs, "title")
    desc = col(jobs, "description")
    company = col(jobs, "company")
    industry_col = col(jobs, "company_industry")
    level_col = col(jobs, "job_level")
    type_col = col(jobs, "job_type")

    # 1. Exclude List Filter
    excludes = c.get("exclude_keywords", [])
    exclude_found = title.apply(lambda t: whole_word(t, excludes))
    keep = exclude_found == ""
    log.append(f"Dropped for excluded title keywords (e.g. retail): {int((~keep).sum())}")

    # 2. Mandatory Role Keyword Filter
    roles = c.get("role_keywords", [])
    role_found = title.apply(lambda t: word_start(t, roles))
    role_keep = role_found != ""
    log.append(f"Dropped for missing role keyword in title: {int((keep & ~role_keep).sum())}")
    keep = keep & role_keep

    # 3. Seniority Filter
    sen_words = c.get("seniority_keywords", [])
    sen_found = title.apply(lambda t: whole_word(t, sen_words))
    sen_keep = sen_found != ""
    seniority = sen_found.where(level_col == "", level_col)
    log.append(f"Dropped for seniority: {int((keep & ~sen_keep).sum())}")
    keep = keep & sen_keep

    # 4. Industry Check
    ind_words = c.get("industries", [])
    blob = title + " " + desc + " " + industry_col
    ind_found = blob.apply(lambda t: word_start(t, ind_words))
    has_info = (desc != "") | (industry_col != "")
    verified = ind_found != ""
    ind_keep = verified | ~has_info
    industry_status = pd.Series("Unverified", index=jobs.index)
    industry_status[verified] = "Verified"
    industry = industry_col.where(industry_col != "", ind_found)
    log.append(f"Dropped for industry: {int((keep & ~ind_keep).sum())}")
    keep = keep & ind_keep

    # Priority Companies check
    pri_names = c.get("priority_companies", [])
    is_pri = company.apply(lambda v: is_priority(v, pri_names))
    priority = is_pri.apply(lambda x: "Yes" if x else "")

    # Calculate Fit Score
    fit_score = pd.Series(0, index=jobs.index)
    fit_score += is_pri.astype(int) * 3           # +3 for priority company
    fit_score += verified.astype(int) * 2         # +2 for verified telecom industry
    fit_score += (title != "").astype(int) * 2    # +2 base match

    out = pd.DataFrame(
        {
            "Fit Score": fit_score,
            "Priority": priority,
            "Title": title,
            "Company": company,
            "Location": col(jobs, "location"),
            "Employment type": type_col,
            "Industry": industry,
            "Industry check": industry_status,
            "Seniority level": seniority,
            "Site": col(jobs, "site"),
            "Date posted": col(jobs, "date_posted"),
            "Job link": col(jobs, "job_url"),
        }
    )[keep]

    # Sort by Fit Score descending, then by Date Posted descending
    out["_date"] = pd.to_datetime(out["Date posted"], errors="coerce")
    out = out.sort_values(["Fit Score", "_date"], ascending=[False, False])
    out = out.drop(columns=["_date"]).reset_index(drop=True)

    log.append(f"Final jobs kept: {len(out)}")
    today = date.today().isoformat()
    out.to_csv(f"results/jobs_{today}.csv", index=False, encoding="utf-8-sig")
    out.to_csv("results/latest.csv", index=False, encoding="utf-8-sig")

    with open("results/run_log.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(log))
    print("\n".join(log))

if __name__ == "__main__":
    main()
