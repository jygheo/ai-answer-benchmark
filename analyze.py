"""
Analyze collected LLM bench runs.

  python analyze.py check        good_culture --wave 2026_Q4   quality gate
  python analyze.py parse        good_culture --wave 2026_Q4   brand mentions + citations
  python analyze.py review       good_culture --wave 2026_Q4   human review of unknown brands/domains
  python analyze.py judge-api    good_culture --wave 2026_Q4   Gemini judge, automatic
  python analyze.py judge-ui     good_culture --wave 2026_Q4   Gemini judge via logged-in Chrome UI

Judge notes (start a wave with these; changing them mid-wave changes the prompts):
  * item ids are "i" + hash, so they can never be mistaken for numbers
  * shopping-card clutter (prices, star ratings, distances, link targets) is stripped before the judge sees an answer
  * Gemini output is schema-constrained; load_json_text also repairs bare item_id values
  * quotes are checked with a markdown/punctuation-tolerant match and flagged (quote_verified), not used to reject items
  * judge QA (unverified quotes, suspect items, top_pick count) is printed to the console; no extra CSVs are written
"""
import argparse
import hashlib
import json
import os
import random
import re
import string
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd
import requests
import tldextract
import yaml
from bs4 import BeautifulSoup
from rapidfuzz import fuzz

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------- global config
STOPLIST = {
    "pros", "cons", "best overall", "best for", "price", "taste", "texture",
    "ingredients", "nutrition", "calcium", "protein", "sodium", "calories",
    "fat", "carbs", "sugar", "target", "walmart", "whole foods", "amazon",
    "kroger", "safeway", "aldi", "costco", "cottage cheese", "brand", "cups",
    "tubs", "dairy", "cheese", "curd", "curds", "low fat", "fat free", "whole milk",
}

# Aliases (lowercase) that are also ordinary words. Matched case-sensitively and
# only when a product word is nearby. A full name such as "Friendship Dairies" is
# its own alias and is not affected.
AMBIGUOUS_ALIASES = {"hood", "friendship", "365"}
PRODUCT_WORDS = ("cottage", "cheese", "brand", "dairy")
STORE_BRAND_ALIASES = ["store brand", "store-brand", "generic", "private label"]

# Cited links that are redirect wrappers (resolved with one HEAD request, cached).
WRAPPER_HOSTS = {"vertexaisearch.cloud.google.com"}

# Page furniture that is not part of the written answer (ChatGPT cards / chips).
NON_PROSE_SELECTORS = (
    "[data-assistant-product-list], [data-assistant-grouped-webpages], "
    "[data-content-reference-type], [data-assistant-content-reference], "
    "[data-assistant-sources-trigger]"
)

JUDGED_STAGES = {"discovery", "attributes", "personas", "comparisons", "objections"}
BATCH_SIZE = 6
MAX_API_ROUNDS = 3

# bundled public-suffix snapshot only: no network
_TLD = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)

# ================================================================ loading
def load_evaluation(name):
    folder = Path(name) if Path(name).is_dir() else ROOT / "evaluations" / name
    if not folder.is_dir():
        raise FileNotFoundError(f"No evaluation folder '{name}' found.")

    def read(p):
        return (yaml.safe_load(open(p, encoding="utf-8")) or {}) if p.exists() else {}

    ev = dict(
        name=folder.name, folder=folder,
        profile=read(folder / "brand_profile.yaml"),
        learned=read(folder / "learned.yaml"),
        engines=read(ROOT / "engines.yaml"),
    )
    apply_learned(ev)
    return ev

def apply_learned(ev):
    """Merge learned.yaml into the profile. ev['learned'] stays the raw file."""
    P, L = ev["profile"], ev["learned"]
    P.setdefault("brands", {})
    P.setdefault("domains", {})
    for k, v in (L.get("brands") or {}).items():
        P["brands"][k] = list(P["brands"].get(k) or []) + list(v or [])
    for k, v in (L.get("domains") or {}).items():
        P["domains"][k] = list(P["domains"].get(k) or []) + list(v or [])
    STOPLIST.update(str(n).lower() for n in (L.get("ignore_names") or []))
    # sites already classified in `domains` are publishers/retailers, not brands: "healthline", "target", ...
    for lst in P["domains"].values():
        STOPLIST.update(str(d).lower().split(".")[0] for d in (lst or []))


def wave_dirs(ev, wave):
    d = ev["folder"] / "results" / wave
    return d, d / "tables", d / "judge"

def load_runs(ev, wave):
    d, _, _ = wave_dirs(ev, wave)
    answers_file, plan_file = d / "answers.jsonl", d / "plan.csv"
    if not answers_file.exists() or not plan_file.exists():
        raise FileNotFoundError(f"Missing answers.jsonl or plan.csv in {d}")

    runs = [json.loads(line) for line in open(answers_file, encoding="utf-8") if line.strip()]
    df = pd.DataFrame(runs).drop_duplicates(subset=["run_id"], keep="first")
    plan = pd.read_csv(plan_file)
    # answers.jsonl already carries test/stage/group; the plan is authoritative.
    df = df.drop(columns=[c for c in ("test", "stage", "group") if c in df.columns])
    return df.merge(plan[["run_id", "test", "stage", "group"]], on="run_id", how="left")


def as_list(x):
    return x if isinstance(x, list) else []

def get_turns(run):
    turns = run.get("turn_results")
    if isinstance(turns, list) and turns:
        return turns
    fa = run.get("final_answer")
    return [{
        "user": "",
        "answer": fa if isinstance(fa, str) else "",
        "html": run.get("final_answer_html") if isinstance(run.get("final_answer_html"), str) else "",
        "cited_urls": as_list(run.get("cited_urls")),
    }]


def text_for(run, scope="final"):
    turns = get_turns(run)
    if scope == "final":
        return turns[-1]["answer"]
    if scope == "first":
        return turns[0]["answer"]
    if scope == "all":
        return "\n\n".join(t["answer"] for t in turns)
    raise ValueError(f"Unknown scope: {scope}")

def valid_runs(ev, wave, df):
    """Drop runs excluded by `check`."""
    _, tables, _ = wave_dirs(ev, wave)
    q = tables / "quality_report.csv"
    if not q.exists():
        print("WARNING: no quality_report.csv, so nothing is excluded. Run `check` first.")
        return df
    qr = pd.read_csv(q)
    bad = set(qr.loc[qr["excluded"].astype(bool), "run_id"])
    return df[~df["run_id"].isin(bad)]


# ================================================================ text + url helpers
def norm_text(t):
    if not t:
        return ""
    t = re.sub(r"[’‘`]", "'", str(t)).replace("&", "and")
    return re.sub(r"\s+", " ", t).strip()


def url_host(u):
    h = urlparse(u).netloc.lower().split(":")[0]
    return h[4:] if h.startswith("www.") else h


def reg_domain(host):
    e = _TLD(host)
    return (getattr(e, "top_domain_under_public_suffix", None) or e.registered_domain or host).lower()


def own_hosts(ev):
    """engine -> its own site host, from engines.yaml. Links back to the chatbot itself are not sources."""
    return {name: url_host(cfg.get("url", "")) for name, cfg in (ev.get("engines") or {}).items()}

def is_own(url, engine, hosts):
    h, own = url_host(url), hosts.get(engine, "")
    return bool(own) and (h == own or h.endswith("." + own))


def external_urls(urls, engine, hosts):
    return list(dict.fromkeys(u for u in as_list(urls) if not is_own(u, engine, hosts)))


def google_surface(url):
    """Links to Google's own result pages are not sources.
    Returns 'google_shopping' (product cards), 'search_page', or None."""
    p = urlparse(url)
    if reg_domain(url_host(url)) != "google.com":
        return None
    q = parse_qs(p.query)
    if url_host(url) == "shopping.google.com" or (p.path == "/search" and ("prds" in q or "oshop" in q.get("ibp", []))):
        return "google_shopping"
    return "search_page" if p.path == "/search" else None

def source_urls(urls, engine, hosts):
    """Real cited sources: not the chatbot's own site, not Google result/shopping pages."""
    return [u for u in external_urls(urls, engine, hosts) if not google_surface(u)]


def is_wrapper(url):
    h = url_host(url)
    return h in WRAPPER_HOSTS or (h == "google.com" and urlparse(url).path == "/url")


def resolve_url(url, cache):
    """Only redirect wrappers are followed. Returns (final_url, resolved_ok)."""
    if not is_wrapper(url):
        return url, True
    if cache.get(url):
        return cache[url], True
    hdr = {"User-Agent": "Mozilla/5.0"}
    final = None
    try:
        final = requests.head(url, allow_redirects=True, timeout=5, headers=hdr).url
        if is_wrapper(final):
            r = requests.get(url, allow_redirects=True, timeout=5, headers=hdr, stream=True)
            final = r.url
            r.close()
    except Exception:
        final = None
    if final and not is_wrapper(final):
        cache[url] = final
        return final, True
    return url, False

# ================================================================ Step 4: check
REFUSAL = re.compile(
    r"i can't help with|i'm unable to|i am unable to|can't browse|cannot browse|i cannot fulfill",
    re.IGNORECASE,
)


def citation_coverage(df, ev):
    hosts = own_hosts(ev)
    rows = []
    for _, r in df.iterrows():
        n = len(source_urls(get_turns(r)[-1].get("cited_urls"), r["engine"], hosts))
        rows.append((r["engine"], n == 0))
    cov = pd.DataFrame(rows, columns=["engine", "zero"]).groupby("engine")["zero"].agg(["mean", "count"])
    print("\nShare of runs with zero cited sources (final turn; own-site and Google search/shopping links not counted):")
    for eng, row in cov.iterrows():
        note = "   <- selectors probably not capturing sources; leave out of citation conclusions" if row["mean"] >= 0.9 else ""
        print(f"  {eng:12s} {row['mean']:.0%}  (n={int(row['count'])}){note}")

def cmd_check(a):
    ev = load_evaluation(a.evaluation)
    df = load_runs(ev, a.wave)
    hosts = own_hosts(ev)

    flags = []
    for _, row in df.iterrows():
        final = norm_text(text_for(row, "final"))
        empty = len(final) < 40
        refusal = bool(REFUSAL.search(final))
        stale = bool(row.get("final_turn_is_new", True) == False)  # noqa: E712
        no_cit = len(source_urls(get_turns(row)[-1].get("cited_urls"), row["engine"], hosts)) == 0
        names = [n for n, v in (("empty_final", empty), ("refusal", refusal),
                                ("stale_final", stale), ("no_citations", no_cit)) if v]
        flags.append(dict(run_id=row["run_id"], engine=row["engine"], stage=row["stage"],
                          flags="|".join(names), empty_final=empty, refusal=refusal,
                          stale_final=stale, no_citations=no_cit,
                          excluded=empty or refusal or stale))

    f = pd.DataFrame(flags)
    _, tables, _ = wave_dirs(ev, a.wave)
    tables.mkdir(parents=True, exist_ok=True)
    f.to_csv(tables / "quality_report.csv", index=False)

    print(f"Loaded {len(df)} distinct runs. Wrote {tables / 'quality_report.csv'}")
    print(f.groupby(["engine", "stage"], dropna=False).agg(
        total=("run_id", "count"), excluded=("excluded", "sum"), empty=("empty_final", "sum"),
        refusal=("refusal", "sum"), stale=("stale_final", "sum")).to_string())
    citation_coverage(df, ev)

# ================================================================ Step 5: parse
def build_known_brands_map(brands):
    """alias (normalized, lowercase) -> canonical. Used to map names the judge returns."""
    m = {}
    for canonical, aliases in brands.items():
        names = [canonical] + list(aliases or [])
        if canonical == "store brand":
            names += STORE_BRAND_ALIASES
        for n in names:
            m[norm_text(n).lower()] = canonical
    return m


def build_matchers(brands):
    """[(compiled regex, canonical, ambiguous)]; ambiguous aliases are case-sensitive."""
    out = []
    for canonical, aliases in brands.items():
        names = [canonical] + list(aliases or [])
        if canonical == "store brand":
            names += STORE_BRAND_ALIASES
        for n in dict.fromkeys(norm_text(x) for x in names if x):
            amb = n.lower() in AMBIGUOUS_ALIASES
            plural = "s?" if canonical == "store brand" else ""   # "store brands", "private labels"
            pat = re.compile(rf"(?<!\w){re.escape(n)}{plural}(?!\w)", 0 if amb else re.IGNORECASE)
            out.append((pat, canonical, amb))
    return out

def match_spans(text, matchers):
    """Non-overlapping (start, end, canonical). Longest alias wins, so 'Daisy Brand' is one mention."""
    found = []
    for pat, canonical, amb in matchers:
        for m in pat.finditer(text):
            if amb:
                ctx = text[max(0, m.start() - 40): m.end() + 40].lower()
                if not any(w in ctx for w in PRODUCT_WORDS):
                    continue
            found.append((m.start(), m.end(), canonical))
    found.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    kept, last_end = [], -1
    for s in found:
        if s[0] >= last_end:
            kept.append(s)
            last_end = s[1]
    return kept


def strip_echo(answer, question):
    na, nq = norm_text(answer), norm_text(question)
    return re.sub(re.escape(nq), " ", na, flags=re.IGNORECASE) if nq else na

def text_list_rank(text, pos):
    """Fallback when the text still has markdown list markers."""
    line_start = text[:pos][text[:pos].rfind("\n") + 1:]
    m = re.match(r"^\s*(?:(\d+)[\.\)]|[-*])\s+", line_start)
    if not m:
        return None
    if m.group(1):
        return int(m.group(1))
    return len(re.findall(r"(?:^|\n)\s*[-*]\s+", text[:pos])) + 1


def html_list_ranks(html, matchers):
    """brand -> 1-based position of the first top-level <li> whose own text names it.
    inner_text() normally drops list numbers, so the rank has to come from the html."""
    if not html:
        return {}
    soup = BeautifulSoup(html, "html.parser")
    for el in soup.select(NON_PROSE_SELECTORS):
        el.decompose()
    ranks = {}
    for li in soup.find_all("li"):
        if li.find_parent("li") is not None:
            continue
        siblings = li.parent.find_all("li", recursive=False) if li.parent else [li]
        idx = next((i for i, s in enumerate(siblings, 1) if s is li), None)
        own = BeautifulSoup(str(li), "html.parser")
        for nested in own.find_all(["ul", "ol"]):
            nested.decompose()
        for _, _, c in match_spans(norm_text(own.get_text(" ")), matchers):
            ranks.setdefault(c, idx)
    return ranks

def find_known_brands(answer, question, matchers, html=None):
    text = strip_echo(answer, question)
    by = {}
    for s, e, c in match_spans(text, matchers):
        by.setdefault(c, []).append(s)
    ranks = html_list_ranks(html, matchers)
    out = []
    for c, pos in by.items():
        rank = ranks.get(c)
        if rank is None:
            rank = text_list_rank(text, pos[0])
        out.append(dict(brand=c, position=pos[0], list_rank=rank,
                        mentions_in_answer=len(pos), known=True))
    return out


# Words that make a phrase "not a brand" when the whole phrase is made of them
GENERIC_WORDS = set("""
pros cons best top overall key takeaway takeaways summary bottom line final verdict quick tip tips note notes
why how what when which who the a an this that these those it its if for with without and or but in on at of to
is are can may might should would could will your you our we they their about also more most less least
high higher low lower good better healthy health protein proteins calcium fat fats sodium sugar sugars carbs
carbohydrates calories calorie serving servings price prices cost value taste tastes flavor flavors texture
creamy thick ingredients ingredient nutrition nutritional benefits benefit downsides drawbacks option options
choice choices brands brand products product store stores grocery groceries retailer retailers organic natural
plain regular original classic small large curd curds cottage cheese yogurt greek dairy milk cream whole
lactose free live active cultures culture probiotics gut digestion weight loss muscle diet snack snacks
breakfast lunch dinner recipe recipes availability available widely easy easiest affordable cheapest budget
premium lowest highest cups cup tubs tub container containers packaging mix mixed fruit fruits vegetables
per fact facts info information guide comparison review reviews list lists chart table
toddler toddlers kids adults people buyers shoppers consumers users customers glp gmo fda usda rda
""".split())
STARTER_WORDS = set("""
best top why how what when which who key quick bottom final overall summary pros cons note tip tips look check
try choose avoid buy consider bonus the this these that if for with without and or but also see
""".split())
CONNECTORS = {"and", "of", "by", "de", "la"}
NUMERIC_JUNK = re.compile(r"\d+\s?(?:g|mg|oz|%|cups?|calories|grams|grams?)\b|[%$@#]|https?:|www\.", re.IGNORECASE)

def looks_like_brand(c):
    """Shape test only. Frequency is handled separately."""
    words = c.split()
    if not (3 <= len(c) <= 30) or not (1 <= len(words) <= 3):
        return False
    if not re.search(r"[A-Za-z]", c) or NUMERIC_JUNK.search(c):
        return False
    if c.lower() in STOPLIST or words[0].lower() in STARTER_WORDS:
        return False
    # brand names are Title Case: every word starts uppercase/digit (connectors aside)
    if any(not (w[0].isupper() or w[0].isdigit() or w.lower() in CONNECTORS) for w in words):
        return False
    core = [w.lower().strip("'s").strip("'") for w in words if w.lower() not in CONNECTORS]
    if not core or all(w in GENERIC_WORDS for w in core):
        return False
    # sentence fragments: ends in a verb-ish or stop word
    if words[-1].lower() in GENERIC_WORDS and len(words) > 1 and words[0].lower() in GENERIC_WORDS:
        return False
    return True

def clean_candidate(c):
    c = norm_text(c).strip(" \t*:-–—•.,;()[]\"'")
    return re.sub(r"^the\s+", "", c, flags=re.IGNORECASE)


def find_candidate_brands(html, text, matchers):
    raw = []
    if html:
        soup = BeautifulSoup(html, "html.parser")
        for el in soup.select(NON_PROSE_SELECTORS):
            el.decompose()
        for tag in soup.find_all(["b", "strong", "h3", "h4"]):
            raw.append(tag.get_text(" ", strip=True))
        for tr in soup.find_all("tr"):
            td = tr.find(["td", "th"])
            if td:
                raw.append(td.get_text(" ", strip=True))
        for li in soup.find_all("li"):
            full = norm_text(li.get_text(" ", strip=True))
            lead = re.split(r":| - | – | — ", full, maxsplit=1)[0]
            if len(lead) < len(full):
                raw.append(lead)
    word = r"[A-Z][\w'&.-]*"
    raw += re.findall(rf"\b({word}(?: {word}){{0,2}}) (?:cottage cheese|brand|cups?|tubs?)\b", norm_text(text))

    out = []
    for c in dict.fromkeys(clean_candidate(x) for x in raw):
        if looks_like_brand(c) and not match_spans(c, matchers):
            out.append(c)
    return out


def sentence_with(text, cand):
    for s in re.split(r"(?<=[.!?])\s+|\n+", text):
        if cand.lower() in s.lower():
            return s.strip()[:200]
    return norm_text(text)[:120]

def classify_domain(url, target, brand_domains, category_domains):
    """url is already resolved. Returns (domain, domain_type, classified_by)."""
    surface = google_surface(url)
    if surface:
        return "google.com", surface, "rule_google_surface"
    domain = reg_domain(url_host(url))
    for brand, doms in brand_domains.items():
        if domain in doms:
            return domain, ("owned" if brand == target else "competitor_owned"), f"matches_{brand}"
    for dtype, lst in category_domains.items():
        if domain in {str(d).lower() for d in (lst or [])}:
            return domain, dtype, "config"
    if domain.endswith(".gov") or domain.endswith(".edu"):
        return domain, "health_authority", "rule_gov_edu"
    if any(x in domain for x in ("reddit", "tiktok", "youtube", "facebook")):
        return domain, "community", "rule_social"
    return domain, "unclassified", "no_match"

def group_candidates(rows, known):
    ex = pd.DataFrame(rows).groupby("candidate").agg(
        count=("run_id", "size"), runs=("run_id", lambda x: set(x)),
        examples=("example", lambda x: list(x))).reset_index()
    out, used = [], set()
    for i, ra in ex.iterrows():
        if i in used:
            continue
        used.add(i)
        name, cnt, runs, exs = ra["candidate"], ra["count"], set(ra["runs"]), list(ra["examples"])
        for j, rb in ex.iterrows():
            if j in used:
                continue
            if fuzz.token_set_ratio(name.lower(), rb["candidate"].lower()) >= 90:
                cnt += rb["count"]; runs |= rb["runs"]; exs += rb["examples"]; used.add(j)
        best, score = None, 0
        for k in known:
            s = fuzz.token_set_ratio(name.lower(), k.lower())
            if s >= 80 and s > score:
                best, score = k, s
        out.append(dict(candidate=name, count=cnt, distinct_runs=len(runs),
                        example_sentences=" | ".join(list(dict.fromkeys(exs))[:3]),
                        suggested_known=best))
    return pd.DataFrame(out).sort_values("count", ascending=False)

def cmd_parse(a):
    ev = load_evaluation(a.evaluation)
    df = valid_runs(ev, a.wave, load_runs(ev, a.wave))
    d, tables, _ = wave_dirs(ev, a.wave)
    tables.mkdir(parents=True, exist_ok=True)

    brands = ev["profile"]["brands"]
    matchers = build_matchers(brands)
    target = ev["profile"].get("target")
    brand_domains = {b: [str(x).lower() for x in (al or []) if "." in str(x) and " " not in str(x)]
                     for b, al in brands.items()}
    category_domains = ev["profile"].get("domains", {})
    hosts = own_hosts(ev)

    cache_path = d / "redirects.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    mentions, unknown_rows, cites = [], [], []
    n_own = n_unresolved = 0
    for _, row in df.iterrows():
        for t_idx, turn in enumerate(get_turns(row), 1):
            ans, html = turn.get("answer", ""), turn.get("html", "")
            for m in find_known_brands(ans, turn.get("user", ""), matchers, html):
                mentions.append(dict(run_id=row["run_id"], turn=t_idx, **m))
            for c in find_candidate_brands(html, ans, matchers):
                unknown_rows.append(dict(candidate=c, run_id=row["run_id"], example=sentence_with(ans, c)))

            for url in as_list(turn.get("cited_urls")):
                final, ok = resolve_url(url, cache)
                if is_own(final, row["engine"], hosts):
                    n_own += 1
                    continue
                if not ok:
                    n_unresolved += 1
                    cites.append(dict(run_id=row["run_id"], engine=row["engine"], turn=t_idx, url=url,
                                      resolved_url=url, domain=reg_domain(url_host(url)),
                                      domain_type="unclassified", classified_by="unresolved_wrapper"))
                    continue
                dom, dtype, by = classify_domain(final, target, brand_domains, category_domains)
                cites.append(dict(run_id=row["run_id"], engine=row["engine"], turn=t_idx, url=url,
                                  resolved_url=final, domain=dom, domain_type=dtype, classified_by=by))

    cache_path.write_text(json.dumps(cache, indent=2))
    pd.DataFrame(mentions).to_csv(tables / "mentions.csv", index=False)
    cdf = pd.DataFrame(cites)
    cdf.to_csv(tables / "citations.csv", index=False)

    ucols = ["candidate", "count", "distinct_runs", "example_sentences", "suggested_known"]
    grouped = group_candidates(unknown_rows, brands) if unknown_rows else pd.DataFrame(columns=ucols)
    keep = (grouped["count"] >= a.min_count) & (grouped["distinct_runs"] >= a.min_runs)
    grouped[keep].to_csv(tables / "unknown_brands.csv", index=False)
    print(f"unknown brands: {int(keep.sum())} kept (>= {a.min_count} mentions in >= {a.min_runs} runs), "
          f"{int((~keep).sum())} set aside as below threshold")

    if not cdf.empty:
        un = cdf[cdf["domain_type"] == "unclassified"]
        un.groupby("domain").agg(
            citations=("url", "size"), engines=("engine", lambda x: ",".join(sorted(set(x)))),
            example_urls=("url", lambda x: " | ".join(list(dict.fromkeys(x))[:3])),
        ).reset_index().sort_values("citations", ascending=False).to_csv(tables / "unknown_domains.csv", index=False)

    n_g = int(cdf["domain_type"].isin(["google_shopping", "search_page"]).sum()) if not cdf.empty else 0
    print(f"{n_g} of the citations are Google search/shopping links (kept in citations.csv, excluded from source counts).")
    print(f"{len(df)} runs parsed: {len(mentions)} brand mentions, {len(cites)} citations "
          f"({n_own} links back to the chatbot's own site dropped, {n_unresolved} unresolved redirect wrappers).")
    citation_coverage(df, ev)



# ================================================================ Step 6: review
DOMAIN_CHOICES = {"r": "retailer", "e": "editorial", "h": "health_authority", "c": "community", "o": "other"}


def cmd_review(a):
    ev = load_evaluation(a.evaluation)
    _, tables, _ = wave_dirs(ev, a.wave)
    learned_path = ev["folder"] / "learned.yaml"
    learned = ev["learned"]
    learned.setdefault("brands", {})
    learned.setdefault("ignore_names", [])
    learned.setdefault("domains", {})

    def save():
        with open(learned_path, "w", encoding="utf-8") as f:
            yaml.dump(learned, f, sort_keys=False, allow_unicode=True)

    known = {b.lower(): b for b in ev["profile"]["brands"]}   # profile + learned
    seen_names = {n.lower() for n in learned["ignore_names"]} | {k.lower() for k in learned["brands"]}
    seen_names |= {str(x).lower() for al in learned["brands"].values() for x in (al or [])}

    frames = []
    for fn in ("unknown_brands.csv", "judge_unknown_brands.csv"):
        p = tables / fn
        if p.exists():
            x = pd.read_csv(p)
            x["example_sentences"] = x.get("example_sentences", "").fillna("") if "example_sentences" in x else ""
            frames.append(x[["candidate", "count", "example_sentences"]])
    if not frames:
        print("No unknown_brands.csv found. Run `parse` first.")
        return
    ub = (pd.concat(frames).groupby("candidate", as_index=False)
          .agg(count=("count", "sum"), example_sentences=("example_sentences", "first"))
          .sort_values("count", ascending=False))
    ub = ub[ub["count"] >= a.min_count]

    print("--- Unknown brands ---   (q = save and stop)")
    for _, row in ub.iterrows():
        cand = str(row["candidate"])
        if cand.lower() in seen_names or cand.lower() in STOPLIST:
            continue
        print(f"\n{cand}   (count {row['count']})\n  {row['example_sentences']}")
        ch = input("[a]dd as new  [m]erge into known  [n]ot a brand  [s]kip  [q]uit: ").strip().lower()
        if ch == "q":
            save(); return
        if ch == "a":
            learned["brands"][cand] = []
        elif ch == "m":
            while True:
                tgt = input("  merge into which known brand (exact name, blank to skip): ").strip()
                if not tgt:
                    break
                if tgt.lower() in known:
                    learned["brands"].setdefault(known[tgt.lower()], []).append(cand)
                    break
                print("  not a known brand. Known:", ", ".join(known.values()))
        elif ch == "n":
            learned["ignore_names"].append(cand)
        save()

    dfile = tables / "unknown_domains.csv"
    if dfile.exists():
        have = {str(d).lower() for lst in list(ev["profile"]["domains"].values()) for d in (lst or [])}
        print("\n--- Unknown domains ---")
        for _, row in pd.read_csv(dfile).sort_values("citations", ascending=False).iterrows():
            dom = str(row["domain"])
            if dom.lower() in have:
                continue
            ex = " | ".join(u[:90] + ("..." if len(u) > 90 else "") for u in str(row["example_urls"]).split(" | "))
            print(f"\n{dom}   (citations {row['citations']}, engines {row['engines']})\n  {ex}")
            ch = input("[r]etailer [e]ditorial [h]ealth [c]ommunity [o]ther [s]kip [q]uit: ").strip().lower()
            if ch == "q":
                break
            if ch in DOMAIN_CHOICES:
                learned["domains"].setdefault(DOMAIN_CHOICES[ch], []).append(dom)
                save()
    save()
    print("\nSaved learned.yaml. Rerun `parse`.")

# ================================================================ Step 7: judge
JUDGE_INSTRUCTION = """You are scoring chatbot answers about {category}. Use ONLY the answer text. Do not use outside knowledge.
For each ITEM return one JSON object. Output a single JSON array and nothing else.
item_id must be copied exactly from the ITEM header and written as a JSON string in double quotes, e.g. "item_id": "i98eb80".

Fields by TYPE:
- ranking:          item_id, brands[], unlisted_brands[], refusal
- head_to_head:     item_id, brands[], winner, unlisted_brands[], refusal
- leading_question: item_id, brands[], agrees_with_user, refusal
- objection:        item_id, brands[], themes[], refusal

brands[]: one entry for every brand or product maker the answer names, from KNOWN BRANDS or not:
  {"name": str, "role": "top_pick|recommended|listed|discouraged", "sentiment": -2..2, "quote": str}
  role:
    top_pick     = the answer explicitly chooses this as its single best or first choice for the user's question
                   ("I'd get X", "X is the best", "go with X"). At most ONE top_pick per answer.
                   A list of options is NOT a choice, even if one option is listed first: use recommended for those,
                   and give the answer no top_pick at all.
    recommended  = suggested as a good option, but not the single choice
    listed       = named without endorsement or criticism
    discouraged  = advised against, or described as worse than an alternative
  sentiment: an integer, how the answer talks about THAT brand: -2 very negative, 0 neutral, +2 very positive.
    Score each brand on what the answer says about it; do not give every brand the same score by default.
  quote: copy 5-25 consecutive words EXACTLY, character for character, from ANSWER TO SCORE only
    (never from CONTEXT or USER MESSAGES). Do not paraphrase, summarize, merge sentences or use "...".
unlisted_brands[]: names of brands NOT in KNOWN BRANDS that the answer names (strings only)
winner (head_to_head only): the brand name the answer prefers overall for the user's question, or "tie", or "none"
agrees_with_user (leading_question only): "yes" | "partly" | "no"
themes[] (objection only), pick from: price, taste, texture, sodium, ingredients, availability, nutrition, other
refusal: true if the answer declines or gives no usable answer
If the answer does not mention a KNOWN BRAND, do not include it.
"""


def judge_instruction(category):
    # not str.format: the text contains literal { } braces
    return JUDGE_INSTRUCTION.replace("{category}", str(category))


ROLES = {"top_pick", "recommended", "listed", "discouraged"}
AGREE = {"yes", "partly", "no"}
THEMES = {"price", "taste", "texture", "sodium", "ingredients", "availability", "nutrition", "other"}

# Passed to Gemini so decoding itself is constrained (item_id is always a string, enums are enforced).
# winner / agrees_with_user / themes are optional because they depend on the item type;
# validate_item enforces the per-type rules.
JUDGE_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "item_id": {"type": "string"},
            "brands": {"type": "array", "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "role": {"type": "string", "enum": sorted(ROLES)},
                    "sentiment": {"type": "integer", "minimum": -2, "maximum": 2},
                    "quote": {"type": "string"},
                },
                "required": ["name", "role", "sentiment", "quote"],
            }},
            "unlisted_brands": {"type": "array", "items": {"type": "string"}},
            "winner": {"type": "string"},
            "agrees_with_user": {"type": "string", "enum": sorted(AGREE)},
            "themes": {"type": "array", "items": {"type": "string", "enum": sorted(THEMES)}},
            "refusal": {"type": "boolean"},
        },
        "required": ["item_id", "brands", "refusal"],
    },
}


def map_type(test_name, stage):
    if stage == "objections":
        return "objection"
    if stage == "comparisons":
        return "leading_question" if "leading" in test_name else "head_to_head"
    return "ranking"


# ---- answer cleaning: shopping cards (prices, ratings, distances, link targets) are noise for the judge
RETAILER_LINES = {"walmart", "target", "sprouts", "amazon", "kroger", "safeway", "costco", "whole foods",
                  "aldi", "instacart", "publix", "walgreens", "cvs"}
CARD_LINE = re.compile(r"""^(?:
      \$\s?\d[\d,]*(?:\.\d+)?(?:\s?usd)?          # $6.99
    | \d[\d,]*(?:\.\d+)?\s?usd                     # 4.89USD
    | (?:&|and)\s*more                             # & more
    | \d(?:\.\d)?\s+stars?\s+rating                # 4.4 stars rating
    | \d\.\d                                       # bare rating 4.4
    | \(\s?\d[\d.,]*k?\+?\s?\)                     # (191)  (1.3k+)
    | \(\s?\d[\d.,]*\s?(?:mi|km)\s?\)              # (27.6 mi)
)$""", re.IGNORECASE | re.VERBOSE)

def clean_for_judge(text):
    """Drop product-card lines and turn [text](url) into text. Wording of the prose is untouched."""
    t = re.sub(r"\[([^\]]*)\]\(https?://[^)]*\)", r"\1", text or "")
    kept = []
    for line in t.splitlines():
        s = line.strip()
        if s and (CARD_LINE.match(s) or s.lower() in RETAILER_LINES):
            continue
        kept.append(line.rstrip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


def build_items(ev, df_valid):
    """item_id -> dict(run_id, type, block). Ids are "i" + a hash of run_id: unique, stable across runs,
    and never parsed as a number by the judge."""
    known = ", ".join(ev["profile"]["brands"].keys())
    items, used = {}, set()
    for _, row in df_valid.iterrows():
        if row["stage"] not in JUDGED_STAGES:
            continue
        turns = get_turns(row)
        # stable id derived from run_id, so a rerun produces the same ids and can resume
        digest = hashlib.sha1(str(row["run_id"]).encode("utf-8")).hexdigest()
        n = 6
        while digest[:n] in used:
            n += 1
        used.add(digest[:n])
        iid = "i" + digest[:n]
        itype = map_type(str(row["test"]), row["stage"])
        lines = [f"### ITEM {iid}", f"TYPE: {itype}",
                 "USER MESSAGES: " + " | ".join(t.get("user", "") for t in turns)]
        if len(turns) > 1:
            lines.append("CONTEXT (earlier assistant reply, for reference only, do not score): "
                         + clean_for_judge(turns[0]["answer"])[:1500])
        lines += ["ANSWER TO SCORE:", clean_for_judge(turns[-1]["answer"]), f"KNOWN BRANDS: {known}", ""]
        items[iid] = dict(run_id=row["run_id"], type=itype, block="\n".join(lines))
    return items


def load_json_text(text):
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    # repair: bare item_id values ("item_id": 98eb80 -> "item_id": "98eb80"); quoted ones start with " and don't match
    t = re.sub(r'("item_id"\s*:\s*)([A-Za-z0-9_\-]+)(\s*[,}\n])', r'\1"\2"\3', t)
    try:
        data = json.loads(t, strict=False)
    except json.JSONDecodeError as e:
        try:
            import json_repair            # optional last resort: pip install json-repair
        except ImportError:
            raise e
        data = json_repair.loads(t)
        if not data:
            raise e
    if isinstance(data, dict):
        data = [data] if "item_id" in data else next((v for v in data.values() if isinstance(v, list)), [])
    if isinstance(data, list):
        for it in data:                   # ids that were valid JSON numbers
            if isinstance(it, dict) and "item_id" in it:
                it["item_id"] = str(it["item_id"])
    return data


def quote_key(t):
    """Comparison key: lowercase, markdown, link targets, ellipses and punctuation removed."""
    t = norm_text(t).lower()
    t = re.sub(r"\]\([^)]*\)", " ", t)              # [text](url) -> text
    t = re.sub(r"[*_`#>\[\]]|\.\.\.|…", " ", t)      # markdown, ellipses
    t = re.sub(r"[^a-z0-9% ]+", " ", t)              # punctuation
    return re.sub(r"\s+", " ", t).strip()

def quote_ok(q, keys):
    """True if the quote is found in the answer (any of the comparison keys)."""
    if not isinstance(q, str):
        return False
    k = quote_key(q)
    if not k:
        return False
    if any(k in key for key in keys):
        return True
    # ellipsis: every fragment must appear
    parts = [quote_key(p) for p in re.split(r"\.\.\.|…", q) if quote_key(p)]
    if len(parts) > 1 and any(all(p in key for p in parts) for key in keys):
        return True
    return len(k) >= 15 and any(fuzz.partial_ratio(k, key) >= 92 for key in keys)


def validate_item(item, itype):
    """Structure only. Quotes are not a reason to reject: they are flagged per row (quote_verified)."""
    errs = []
    if not isinstance(item.get("brands"), list):
        return ["brands missing or not a list"]
    if not isinstance(item.get("refusal"), bool):
        errs.append("refusal must be true/false")
    if itype == "head_to_head" and not isinstance(item.get("winner"), str):
        errs.append("winner missing")
    if itype == "leading_question" and item.get("agrees_with_user") not in AGREE:
        errs.append("agrees_with_user must be yes|partly|no")
    if itype == "objection":
        th = item.get("themes")
        if not isinstance(th, list) or any(t not in THEMES for t in th):
            errs.append("themes missing or not in the allowed list")
    if sum(1 for b in item["brands"] if isinstance(b, dict) and b.get("role") == "top_pick") > 1:
        errs.append(">1 top_pick")
    for b in item["brands"]:
        if not isinstance(b, dict) or not isinstance(b.get("name"), str):
            errs.append("brand entry malformed")
            continue
        n = b["name"]
        if b.get("role") not in ROLES:
            errs.append(f"{n}: invalid role {b.get('role')!r}")
        s = b.get("sentiment")
        if isinstance(s, bool) or not isinstance(s, (int, float)) or s != int(s) or not -2 <= s <= 2:
            errs.append(f"{n}: sentiment must be an integer -2..2")
    return errs


def check_judgments(data, expected_ids, meta, df_indexed, brand_map, source):
    """Returns (accepted {item_id: dict(rows, unlisted)}, rejects [dict])."""
    accepted, rejects = {}, []
    if not isinstance(data, list):
        return accepted, [dict(file=source, item_id=i, error="output is not a JSON array") for i in expected_ids or []]
    counts = Counter(it.get("item_id") for it in data if isinstance(it, dict))
    handled = set()
    for item in data:
        if not isinstance(item, dict):
            rejects.append(dict(file=source, item_id=None, error="entry is not an object", item=item))
            continue
        iid = item.get("item_id")
        if iid not in meta or (expected_ids is not None and iid not in expected_ids):
            rejects.append(dict(file=source, item_id=iid, error="unknown or extra item_id", item=item))
            continue
        handled.add(iid)
        if counts[iid] > 1:
            rejects.append(dict(file=source, item_id=iid, error="item_id appears more than once", item=item))
            continue
        run_id = meta[iid]["run_id"]
        run = df_indexed.loc[run_id]
        errs = validate_item(item, meta[iid]["type"])
        if errs:
            rejects.append(dict(file=source, item_id=iid, error="; ".join(errs), item=item))
            continue

        final = text_for(run, "final")
        akeys = (quote_key(final), quote_key(clean_for_judge(final)))

        def canon(name):
            return brand_map.get(norm_text(name).lower())

        win = item.get("winner")
        win = canon(win) or win if isinstance(win, str) else win
        common = dict(item_id=iid, run_id=run_id, type=meta[iid]["type"], engine=run["engine"],
                      test=run["test"], stage=run["stage"],
                      winner=win, agrees_with_user=item.get("agrees_with_user"),
                      themes=" | ".join(item.get("themes") or []), refusal=item["refusal"])
        rows = []
        for b in item["brands"]:
            c = canon(b["name"])
            rows.append(dict(common, brand=c or b["name"], known=c is not None, role=b["role"],
                             sentiment=int(b["sentiment"]), quote=b.get("quote"),
                             quote_verified=quote_ok(b.get("quote"), akeys)))
        if not rows:   # keep winner / agreement / refusal even when no brand was named
            rows.append(dict(common, brand=None, known=None, role=None, sentiment=None,
                             quote=None, quote_verified=None))
        accepted[iid] = dict(rows=rows, unlisted=[u for u in as_list(item.get("unlisted_brands")) if isinstance(u, str)])
    for iid in (expected_ids or []):
        if iid not in handled:
            rejects.append(dict(file=source, item_id=iid, error="missing from output"))
    return accepted, rejects


def judge_diagnostics(jdf, df_indexed, tables):
    """QA of the judge itself: unverified quotes, suspicious items, and a top_pick spot-check sample."""
    named = jdf[jdf["brand"].notna()].copy()
    if named.empty:
        return
    named["quote_verified"] = named["quote_verified"].astype(bool)
    print(f"quotes verified: {named['quote_verified'].mean():.0%} of {len(named)} brand rows")

    per_item = named.groupby("item_id").agg(
        n=("brand", "size"), verified=("quote_verified", "sum"), n_sent=("sentiment", "nunique"))
    suspect = []
    for iid, r in per_item.iterrows():
        reasons = []
        if r["verified"] == 0:
            reasons.append("no_verified_quote")
        if r["n"] >= 3 and r["n_sent"] == 1:
            reasons.append("identical_sentiment")
        if reasons:
            suspect.append(dict(item_id=iid, reasons="|".join(reasons)))
    print(f"suspect items (no verified quote, or identical sentiment on 3+ brands): {len(suspect)} "
          f"(consider re-judging these with a stronger model)")

    rk = named[named["type"] == "ranking"]
    n_rank = jdf[jdf["type"] == "ranking"]["item_id"].nunique()
    tp = rk[rk["role"] == "top_pick"]
    print(f"ranking items with a top_pick: {tp['item_id'].nunique()}/{n_rank}")


def finalize_judgments(ev, wave, accepted, df_indexed):
    """judgments.csv, judge_unknown_brands.csv, judge diagnostics (printed)."""
    _, tables, _ = wave_dirs(ev, wave)
    tables.mkdir(parents=True, exist_ok=True)
    rows = [r for v in accepted.values() for r in v["rows"]]
    jdf = pd.DataFrame(rows)
    to_csv_atomic(jdf, tables / "judgments.csv")

    unk = Counter()
    ex = {}
    for v in accepted.values():
        names = {}
        for r in v["rows"]:
            if r["known"] is False:
                names[r["brand"]] = r["quote"] or ""
        for u in v["unlisted"]:
            names.setdefault(u, "")
        for n, q in names.items():       # once per answer
            unk[n] += 1
            ex.setdefault(n, q)
    brands_l = {b.lower() for b in ev["profile"]["brands"]}
    pd.DataFrame([dict(candidate=k, count=n, example_sentences=ex.get(k, ""))
                  for k, n in unk.most_common() if k.lower() not in brands_l and k.lower() not in STOPLIST],
                 columns=["candidate", "count", "example_sentences"]
                 ).pipe(to_csv_atomic, tables / "judge_unknown_brands.csv")

    mpath = tables / "mentions.csv"
    if mpath.exists() and not jdf.empty:
        mdf = pd.read_csv(mpath)
        diff = []
        for run_id, g in jdf.groupby("run_id"):
            run = df_indexed.loc[run_id]
            last = int(run["n_turns"]) if pd.notna(run.get("n_turns")) else len(get_turns(run))
            judge = set(g.loc[g["known"] == True, "brand"])  # noqa: E712
            regex = set(mdf[(mdf["run_id"] == run_id) & (mdf["turn"] == last)]["brand"]) if not mdf.empty else set()
            diff += [dict(run_id=run_id, brand=b, only_in="judge") for b in sorted(judge - regex)]
            diff += [dict(run_id=run_id, brand=b, only_in="parse") for b in sorted(regex - judge)]
        print(f"mention_disagreement: {len(diff)} brand/run pairs differ between regex and judge")
    elif not mpath.exists():
        print("(run `parse` first to get the mention_disagreement recall check)")
    print(f"Imported {len(accepted)} items -> {len(rows)} rows in {tables / 'judgments.csv'}")
    if not jdf.empty:
        judge_diagnostics(jdf, df_indexed, tables)

def atomic_write(path, text):
    """Write to a temp file, then rename: a crash never leaves a half-written file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)

def to_csv_atomic(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)

def load_checkpoint(path):
    done = {}
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
            done[rec["item_id"]] = rec
        except Exception:
            continue   # a line cut off by an interrupted write is simply skipped
    return done

def append_checkpoint(path, rec):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())

def call_judge(client, config, prompt, model, fallback, raw_dir, tag, tries=6):
    """Returns (parsed list, model used) or (None, None). Backs off on 429/5xx."""
    for attempt in range(tries):
        m = fallback if (fallback and attempt >= tries // 2) else model
        try:
            response = client.chats.create(model=m, config=config).send_message(prompt)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            code = getattr(e, "code", None)
            if code in (400, 401, 403, 404):
                raise RuntimeError(f"Not retrying, request was rejected ({code}): {e}")
            transient = code in (429, 500, 502, 503, 504) or any(
                x in str(e) for x in ("UNAVAILABLE", "RESOURCE_EXHAUSTED", "DEADLINE", "503", "429"))
            wait = (min(120, 5 * 2 ** attempt) + random.uniform(0, 3)) if transient else 3
            print(f"    attempt {attempt + 1}/{tries} with {m} failed: {str(e)[:150]}  (waiting {wait:.0f}s)")
            time.sleep(wait)
            continue
        text = response.text or ""
        try:
            return load_json_text(text), m
        except Exception as e:
            raw_dir.mkdir(parents=True, exist_ok=True)
            (raw_dir / f"{tag}_try{attempt + 1}.txt").write_text(text, encoding="utf-8")
            print(f"    attempt {attempt + 1}/{tries} with {m}: reply was not valid JSON ({e}); "
                  f"raw reply saved in judge/raw_failures/")
            time.sleep(2)
    return None, None


def save_items(judge_dir, items):
    (judge_dir / "items.json").write_text(json.dumps({k: v["block"] for k, v in items.items()}), encoding="utf-8")


def judge_setup(a):
    ev = load_evaluation(a.evaluation)
    df = load_runs(ev, a.wave)
    df_valid = valid_runs(ev, a.wave, df)
    _, tables, judge_dir = wave_dirs(ev, a.wave)
    judge_dir.mkdir(parents=True, exist_ok=True)
    return ev, df, df_valid, judge_dir


def cmd_judge_api(a):
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print("Please install google-genai: pip install google-genai")
        return

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable is not set.")
        return

    client = genai.Client(api_key=api_key)
    model_name = a.model or "gemini-3.5-flash"
    fallback_model = a.fallback or None

    ev, df, df_valid, judge_dir = judge_setup(a)
    items = build_items(ev, df_valid)
    save_items(judge_dir, items)
    meta = {k: dict(run_id=v["run_id"], type=v["type"]) for k, v in items.items()}
    df_indexed = df.set_index("run_id")
    brand_map = build_known_brands_map(ev["profile"]["brands"])

    # Every accepted item is appended here right after its batch is validated.
    ckpt_path = judge_dir / "api_checkpoint.jsonl"
    if a.fresh and ckpt_path.exists():
        ckpt_path.unlink()
    done = {k: v for k, v in load_checkpoint(ckpt_path).items() if k in items}
    accepted = {k: dict(rows=v["rows"], unlisted=v["unlisted"]) for k, v in done.items()}
    models_used = Counter(v.get("model") for v in done.values())
    if done:
        print(f"Resuming: {len(done)}/{len(items)} items already judged, skipping them.")

    config = types.GenerateContentConfig(
        system_instruction=judge_instruction(ev["profile"]["category"]),
        response_mime_type="application/json",
        response_schema=JUDGE_SCHEMA,
        temperature=0,
    )

    last_reject, stopped, fail_streak = {}, False, 0
    try:
        for rnd in range(1, MAX_API_ROUNDS + 1):
            todo = [i for i in items if i not in accepted]
            if not todo or stopped:
                break
            random.shuffle(todo)
            nb = (len(todo) + BATCH_SIZE - 1) // BATCH_SIZE
            print(f"Round {rnd}: sending {len(todo)} items to {model_name} in {nb} batches...")
            for b in range(nb):
                ids = todo[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
                prompt = "\n".join(items[i]["block"] for i in ids)
                print(f"  batch {b + 1}/{nb}  (accepted so far: {len(accepted)}/{len(items)})")
                data, used_model = call_judge(client, config, prompt, model_name, fallback_model,
                                              judge_dir / "raw_failures", f"r{rnd}_b{b + 1}")
                if data is None:
                    fail_streak += 1
                    for i in ids:
                        last_reject[i] = dict(file="api", item_id=i, error="no usable response")
                    if fail_streak >= 3:
                        print("3 batches in a row got no usable response; the service looks down. Stopping.")
                        stopped = True
                        break
                    continue
                fail_streak = 0
                acc, rej = check_judgments(data, set(ids), meta, df_indexed, brand_map, "api")
                for iid, v in acc.items():
                    append_checkpoint(ckpt_path, dict(item_id=iid, model=used_model, **v))
                    models_used[used_model] += 1
                accepted.update(acc)
                for r in rej:
                    if r["item_id"] is not None:
                        last_reject[r["item_id"]] = r
    except KeyboardInterrupt:
        stopped = True
        print("\nInterrupted.")
    except RuntimeError as e:
        stopped = True
        print(f"\nStopped: {e}")

    if stopped:
        print(f"Progress is saved: {len(accepted)}/{len(items)} items in {ckpt_path.name}. "
              f"Run the same command again to continue. judgments.csv was not touched.")
        return

    rejects = [r for i, r in last_reject.items() if i not in accepted]
    atomic_write(judge_dir / "rejects.jsonl", "\n".join(json.dumps(r, default=str) for r in rejects))
    atomic_write(judge_dir / "judge_meta.json", json.dumps(dict(
        judge_model=model_name, models_used=dict(models_used), date=datetime.now().isoformat(timespec="seconds"),
        blinded_to_brand=False, schema_constrained=True, answers_cleaned_of_shopping_cards=True,
        items=len(items), accepted=len(accepted), rejected=len(rejects)), indent=2))
    if len(models_used) > 1:
        print(f"NOTE: this wave was judged by more than one model: {dict(models_used)}")
    print(f"accepted: {len(accepted)}, rejected: {len(rejects)}")
    finalize_judgments(ev, a.wave, accepted, df_indexed)

def cmd_judge_ui(a):
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Please install playwright: pip install playwright")
        return

    import yaml

    ev, df, df_valid, judge_dir = judge_setup(a)
    items = build_items(ev, df_valid)
    save_items(judge_dir, items)
    meta = {k: dict(run_id=v["run_id"], type=v["type"]) for k, v in items.items()}
    df_indexed = df.set_index("run_id")
    brand_map = build_known_brands_map(ev["profile"]["brands"])

    try:
        engines_cfg = yaml.safe_load(open(ROOT / "engines.yaml", encoding="utf-8"))
        cfg = engines_cfg.get("gemini-judge") or engines_cfg["gemini"]
    except Exception as e:
        print(f"Error loading config from engines.yaml: {e}")
        return

    ckpt_path = judge_dir / "ui_checkpoint.jsonl"
    if getattr(a, "fresh", False) and ckpt_path.exists():
        ckpt_path.unlink()
        
    done = {k: v for k, v in load_checkpoint(ckpt_path).items() if k in items}
    accepted = {k: dict(rows=v["rows"], unlisted=v["unlisted"]) for k, v in done.items()}
    models_used = Counter(v.get("model") for v in done.values())
    
    if done:
        print(f"Resuming: {len(done)}/{len(items)} items already judged, skipping them.")

    instruction = judge_instruction(ev["profile"]["category"])
    last_reject, stopped, fail_streak = {}, False, 0
    port = getattr(a, "port", 9223)

    with sync_playwright() as p:
        try:
            browser = p.chromium.connect_over_cdp(f"http://localhost:{port}")
        except Exception:
            print(f"Cannot reach Chrome on port {port}. Ensure you ran `python bench.py chrome --profile regular` first.")
            return

        ctx = browser.contexts[0]
        # Required to read the text after clicking the copy button
        ctx.grant_permissions(["clipboard-read", "clipboard-write"])
        page = ctx.new_page()
        page.set_default_timeout(60000)

        try:
            for rnd in range(1, MAX_API_ROUNDS + 1):
                todo = [i for i in items if i not in accepted]
                if not todo or stopped:
                    break
                random.shuffle(todo)
                nb = (len(todo) + BATCH_SIZE - 1) // BATCH_SIZE
                print(f"Round {rnd}: sending {len(todo)} items via UI in {nb} batches...")
                
                for b in range(nb):
                    ids = todo[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
                    batch_text = "\n".join(items[i]["block"] for i in ids)
                    full_prompt = instruction + "\n\n" + batch_text
                    
                    print(f"  batch {b + 1}/{nb}  (accepted so far: {len(accepted)}/{len(items)})")
                    
                    try:
                        page.goto(cfg["url"], wait_until="domcontentloaded")
                        time.sleep(3) 

                        input_loc = None
                        for sel in cfg["input_box"]:
                            try:
                                loc = page.locator(sel).first
                                loc.wait_for(state="visible", timeout=5000)
                                input_loc = loc
                                break
                            except Exception:
                                continue
                                
                        if not input_loc:
                            raise RuntimeError("Could not find input box using selectors from engines.yaml")
                            
                        input_loc.click()
                        page.keyboard.insert_text(full_prompt)
                        time.sleep(1)
                        page.keyboard.press("Enter")

                        time.sleep(3)
                        def is_busy():
                            return any(page.locator(sel).first.is_visible() for sel in cfg["busy_indicator"])
                        
                        while is_busy():
                            time.sleep(1)
                            
                        time.sleep(2) 
                            
                        # Extract response using the copy button
                        text = ""
                        copy_sel = cfg.get("copy_button", "mat-icon[data-mat-icon-name='copy']")
                        copy_btn = page.locator(copy_sel).last
                        
                        try:
                            # Try to click the copy button and read the clipboard
                            copy_btn.wait_for(state="visible", timeout=3000)
                            copy_btn.click()
                            time.sleep(0.5)
                            text = page.evaluate("navigator.clipboard.readText()")
                        except Exception as e:
                            # Fallback: pull inner_text from the whole response block if the copy button fails
                            print("    (Copy button not found or failed, falling back to block extraction)")
                            reply_loc = None
                            for sel in cfg["reply_blocks"]:
                                if page.locator(sel).count() > 0:
                                    reply_loc = page.locator(sel).last
                                    break
                            if reply_loc:
                                text = reply_loc.inner_text()
                        
                        if not text or not text.strip():
                            raise ValueError("Extracted text was empty.")
                        
                        data = load_json_text(text)
                        used_model = "gemini-ui-flash"
                        
                        fail_streak = 0
                        acc, rej = check_judgments(data, set(ids), meta, df_indexed, brand_map, "ui")
                        for iid, v in acc.items():
                            append_checkpoint(ckpt_path, dict(item_id=iid, model=used_model, **v))
                            models_used[used_model] += 1
                        accepted.update(acc)
                        
                        for r in rej:
                            if r["item_id"] is not None:
                                last_reject[r["item_id"]] = r
                                
                    except Exception as e:
                        print(f"    UI attempt failed: {str(e)[:150]}")
                        fail_streak += 1
                        for i in ids:
                            last_reject[i] = dict(file="ui", item_id=i, error=f"UI failed: {str(e)[:100]}")
                        if fail_streak >= 3:
                            print("3 UI batches failed in a row. Stopping.")
                            stopped = True
                            break
                        time.sleep(5)
                        
        except KeyboardInterrupt:
            stopped = True
            print("\nInterrupted.")
        except Exception as e:
            stopped = True
            print(f"\nStopped: {e}")
        finally:
            page.close()

    if stopped:
        print(f"Progress is saved: {len(accepted)}/{len(items)} items in {ckpt_path.name}. "
              f"Run the same command again to continue. judgments.csv was not touched.")
        return

    rejects = [r for i, r in last_reject.items() if i not in accepted]
    atomic_write(judge_dir / "rejects.jsonl", "\n".join(json.dumps(r, default=str) for r in rejects))
    atomic_write(judge_dir / "judge_meta.json", json.dumps(dict(
        judge_model="gemini-ui-flash", models_used=dict(models_used), date=datetime.now().isoformat(timespec="seconds"),
        blinded_to_brand=False, schema_constrained=True, answers_cleaned_of_shopping_cards=True,
        items=len(items), accepted=len(accepted), rejected=len(rejects)), indent=2))
        
    print(f"accepted: {len(accepted)}, rejected: {len(rejects)}")
    finalize_judgments(ev, a.wave, accepted, df_indexed)

# ================================================================ CLI
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Analyze collected LLM bench runs.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    
    for cmd, func in [("check", cmd_check), ("parse", cmd_parse), ("review", cmd_review),
                      ("judge-api", cmd_judge_api), ("judge-ui", cmd_judge_ui)]:
        p = sub.add_parser(cmd)
        p.add_argument("evaluation")
        p.add_argument("--wave", required=True)
        if cmd in ("parse", "review"):
            p.add_argument("--min-count", type=int, default=3, help="unknown brand must appear this many times (default 3)")
        if cmd == "parse":
            p.add_argument("--min-runs", type=int, default=2, help="...in at least this many distinct runs (default 2)")
        if cmd == "judge-api":
            p.add_argument("--model", default="gemini-3.5-flash", help="Gemini model (default: gemini-3.5-flash)")
            p.add_argument("--fallback", default=None, help="optional second model used for the later retries of a batch")
        if cmd in ("judge-api", "judge-ui"):
            p.add_argument("--fresh", action="store_true", help="ignore saved progress and judge everything again")
        if cmd == "judge-ui":
            p.add_argument("--port", type=int, default=9223, help="Chrome debugging port (default: 9223 for regular profile)")
        p.set_defaults(fn=func)
        
    args = ap.parse_args()
    args.fn(args)