"""
Questions:     evaluations/<name>/prompts.yaml
LLM info:      engines.yaml

Commands:

  python bench.py check  good_culture
      validate the prompts.yaml and brand_profile.yaml files

  python bench.py plan   good_culture --core-only
      write results/<wave>/plan.csv

  python bench.py chrome
      start Chrome with remote debugging, private window

  python bench.py chrome --profile regular
      start Chrome with remote debugging, normal window

  python bench.py probe  [chatgpt, gemini, perplexity]
      test the selectors in engines.yaml on the live site

  python bench.py probe  [chatgpt, gemini, perplexity] --ask "prompt 1" "prompt 2"
      test multiple-turn conversations

  python bench.py run    good_culture --wave 2026_Q4 --limit 12
      prompt the LLMs and save their answers

  python bench.py status good_culture
      show progress

"""
import argparse, itertools, json, math, os, random, re, shutil, subprocess, sys, tempfile, time, traceback
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlparse, urlsplit, urlunsplit

import pandas as pd
import yaml
import json as _json


ROOT = Path(__file__).resolve().parent
STAGES = ["discovery", "attributes", "personas", "comparisons", "objections"]
SPEED = 1.0                      # 1.0 = human pacing; --fast sets 0 (testing only)
WALL_WORDS = ["verify you are human", "just a moment", "captcha", "unusual traffic", "are you a robot"]


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")[:40]


def die(msg):
    sys.exit(f"ERROR: {msg}")


# =============================================================== loading
def load_evaluation(name):
    folder = Path(name) if Path(name).is_dir() else ROOT / "evaluations" / name
    if not folder.is_dir():
        die(f"no evaluation folder '{name}' (expected evaluations/{name}/)")
    if folder.name.startswith("_"):
        die(f"'{folder.name}' is the blank template. Copy it first:  cp -r evaluations/_template evaluations/my_brand")
    read = lambda p: yaml.safe_load(open(p, encoding="utf-8"))
    ev = dict(name=folder.name, folder=folder, profile=read(folder / "brand_profile.yaml"),
              prompts=read(folder / "prompts.yaml"), engines=read(ROOT / "engines.yaml"))
    p = ev["profile"]
    if p["market_leader"] == p["target"]:                       # evaluating the leader itself
        p["market_leader"] = next(b for b in p["brands"] if b != p["target"])
    return ev


def base_vars(ev):
    p = ev["profile"]
    return dict(category=p["category"], target=p["target"], market_leader=p["market_leader"])


# =============================================================== turning a test into concrete prompts
def fill(text, values, where):
    try:
        return text.format(**values)
    except KeyError as e:
        die(f"{where}: unknown placeholder {e}. Available here: {sorted(values)}")


def options(spec):
    """matrix entry -> [(id, text)]. dict form uses its keys as ids; list form slugs the value."""
    if isinstance(spec, dict):
        return list(spec.items())
    return [(slug(v), v) for v in spec]


def expand_test(test, ev):
    """One test from prompts.yaml -> list of {prompt_id, turns}. No hidden rules: just the fields in the yaml."""
    where = f"test '{test['name']}'"
    base = base_vars(ev)
    if "turns" not in test:
        die(f"{where}: missing `turns`")
    if test.get("stage") not in STAGES:
        die(f"{where}: stage '{test.get('stage')}' must be one of {STAGES}")

    if test.get("per_fact_row"):
        combos = [[("row", slug(f"{r['brand']}_{r['product']}"), dict(brand=r["brand"], product=r["product"], serving=r["serving"]))]
                  for r in ev["profile"]["facts"]]
    else:
        matrix = test.get("matrix", {})
        keys = list(matrix)
        combos = [[(k, i, {k: fill(t, base, where)}) for k, (i, t) in zip(keys, pick)]
                  for pick in itertools.product(*[options(matrix[k]) for k in keys])] or [[]]

    out = []
    for combo in combos:
        vals = dict(base)
        for _, _, v in combo:
            vals.update(v)
        suffix = "_".join(i for _, i, _ in combo)
        versions = [(vals, "")]
        if test.get("counterbalance"):
            if "rival" not in vals:
                die(f"{where}: counterbalance needs a `rival` key in matrix")
            versions.append((dict(vals, target=vals["rival"], rival=vals["target"]), "swapped"))
        for v, tag in versions:
            pid = "__".join(x for x in [test["name"], suffix, tag] if x)
            out.append(dict(prompt_id=pid, turns=[fill(t, v, where) for t in test["turns"]]))
    return out


# =============================================================== validation + plan
def all_tests(ev):
    for group, key in (("core", "core_tests"), ("diagnostic", "diagnostic_tests")):
        for t in ev["prompts"].get(key) or []:
            yield group, t


def validate(ev):
    errs, warns, p = [], [], ev["profile"]
    for k in ("target", "market_leader"):
        if p[k] not in p["brands"]:
            errs.append(f"{k} '{p[k]}' is not listed under brands")
    for e in ev["prompts"]["engines"]:
        if e not in ev["engines"]:
            errs.append(f"engine '{e}' not in engines.yaml")
    # for f in p["facts"]:
    #     if f["brand"] not in p["brands"]:
    #         errs.append(f"facts row brand '{f['brand']}' not under brands")
    #     # if f.get("serving") != p["serving"]:
    #     #     warns.append(f"{f['brand']} / {f['product']} is per '{f.get('serving')}', not '{p['serving']}': don't compare per-serving")
    #     if not f.get("source_url"):
    #         warns.append(f"{f['brand']} / {f['product']}: no source_url yet")
    # if not {p["target"], p["market_leader"]} <= {f["brand"] for f in p["facts"]}:
    #     warns.append("facts should include rows for both the target and the market leader")
    names = [t["name"] for _, t in all_tests(ev)]
    if len(names) != len(set(names)):
        errs.append("duplicate test names in prompts.yaml")
    ids = []
    for g, t in all_tests(ev):
        ids += [x["prompt_id"] for x in expand_test(t, ev)]                     
    if len(ids) != len(set(ids)):
        errs.append("two prompts got the same id (change a test name or matrix key)")
    return errs, warns


def build_plan(ev, core_only=False):
    top_engines, rows = ev["prompts"]["engines"], []
    for group, t in all_tests(ev):
        if core_only and group != "core":
            continue
        for pr in expand_test(t, ev):
            for eng in t.get("engines", top_engines):
                searches = ["default"]
                if t.get("also_without_web_search") and ev["engines"][eng].get("web_search_off"):
                    searches.append("off")
                for ws in searches:
                    for rep in range(1, (t.get("reps", 1) if ws == "default" else 1) + 1):
                        rows.append(dict(run_id=f"{pr['prompt_id']}|{eng}|search_{ws}|rep{rep}", prompt_id=pr["prompt_id"],
                                         test=t["name"], stage=t["stage"], group=group, engine=eng, web_search=ws,
                                         rep=rep, n_turns=len(pr["turns"]), turns=json.dumps(pr["turns"], ensure_ascii=False)))
    plan = pd.DataFrame(rows)
    rng, parts = random.Random(ev["prompts"].get("seed", 42)), []
    for _, g in plan.groupby("rep"):     
        idx = list(g.index); rng.shuffle(idx); parts.append(plan.loc[idx])
    return pd.concat(parts).reset_index(drop=True)


def wave_default():
    d = datetime.now()
    return f"{d.year}_Q{(d.month - 1) // 3 + 1}"


def result_paths(ev, wave):
    d = ev["folder"] / "results" / wave
    return d, d / "plan.csv", d / "answers.jsonl", d / "errors.jsonl"


def read_done(answers):
    if not answers.exists():
        return set()
    return {json.loads(l)["run_id"] for l in open(answers, encoding="utf-8") if l.strip()}


def nap(lo, hi):
    time.sleep(random.uniform(lo, hi) * SPEED)


def key_delay():
    d = random.lognormvariate(math.log(0.085), 0.45)     
    if random.random() < 0.03:
        d += random.uniform(0.25, 0.8)    
    return min(d, 1.5)


def human_type(page, text):
    for ch in text:
        page.keyboard.type(ch)
        time.sleep((key_delay() + (random.uniform(0.02, 0.12) if ch in " ,.?" else 0)) * SPEED)


def human_click(page, loc):
    box = loc.bounding_box()
    if not box:
        return loc.click()
    x = box["x"] + box["width"] * random.uniform(0.2, 0.8)
    y = box["y"] + box["height"] * random.uniform(0.3, 0.7)
    page.mouse.move(x, y, steps=random.randint(12, 30))
    nap(0.2, 0.6)
    page.mouse.click(x, y)


# =============================================================== talking to one LLM page
class Blocked(Exception):
    pass


def clean_url(u):
    p = urlsplit(u)
    q = [(k, v) for k, v in parse_qsl(p.query) if not k.lower().startswith("utm_")]
    return urlunsplit((p.scheme, p.netloc, p.path, urlencode(q), ""))


def turn_selectors(cfg):
    return cfg.get("turn_blocks") or [", ".join(cfg["reply_blocks"])]


def snapshot_counts(page, cfg):
    return {s: page.locator(s).count() for s in turn_selectors(cfg)}


def new_turn(page, cfg, before):
    """Highest-priority selector that gained a match since `before`; returns its last element."""
    for s in turn_selectors(cfg):
        loc = page.locator(s)
        if loc.count() > before.get(s, 0):
            return loc.last
    return None


def turn_text(turn):
    return turn.inner_text().strip()


CHIP_SEL = "[data-assistant-content-reference], [data-assistant-sources-trigger], [data-assistant-grouped-webpages]"

def prose_text(turn, cfg):
    sel = (cfg.get("prose_parts") or [None])[0]
    return turn.evaluate("""(el, a) => {
        const chips = [...el.querySelectorAll(a.chips)];
        const old = chips.map(c => c.style.display);
        chips.forEach(c => c.style.display = 'none');
        const nodes = a.sel ? [...el.querySelectorAll(a.sel)] : [el];
        const text = nodes.map(n => n.innerText.trim()).filter(Boolean).join('\\n\\n');
        chips.forEach((c, i) => c.style.display = old[i]);
        return text;
    }""", dict(sel=sel, chips=CHIP_SEL)) or turn_text(turn)

def extract_payloads(turn, attr_sel, attr_name):
    raw = turn.evaluate(
        "(el, a) => [...el.querySelectorAll(a.sel)].map(n => n.getAttribute(a.name))",
        dict(sel=attr_sel, name=attr_name))
    out = []
    for r in raw:
        try:
            out.append(_json.loads(r))
        except Exception:
            pass
    return out

def flatten_sources(payloads):
    urls = []
    for p in payloads:
        for item in (p if isinstance(p, list) else [p]):
            if isinstance(item, dict):
                u = item.get("url") or item.get("href") or item.get("link")
                if u:
                    urls.append(u)
    return list(dict.fromkeys(clean_url(u) for u in urls))



def wait_until_answer_finished(page, cfg, before, timeout=300):
    done_attr = cfg.get("done_attr")
    start, last, since = time.time(), None, None
    stable_for = 3 * SPEED if SPEED else 0.3
    while time.time() - start < timeout:
        time.sleep(1.0 * SPEED if SPEED else 0.05)
        turn = new_turn(page, cfg, before)
        if turn is None:
            continue
        if done_attr and turn.get_attribute(done_attr) is not None:
            time.sleep(1.5 * SPEED)            # let late cards/sources render
            return turn
        text = turn_text(turn)                  # fallback for engines without a done attribute
        busy = any(page.locator(s).first.is_visible() for s in cfg["busy_indicator"])
        if text and text == last and not busy:
            since = since or time.time()
            if time.time() - since >= max(stable_for, 8 * SPEED):
                return turn
        else:
            since = None
        last = text
    raise TimeoutError("answer did not finish")

def send_message(page, cfg, engine, text):
    box = get_input_box(page, cfg, engine)
    before = snapshot_counts(page, cfg)
    human_click(page, box)
    nap(0.3, 0.9)
    human_type(page, text)
    nap(0.5, 1.8)
    page.keyboard.press("Enter")
    turn = wait_until_answer_finished(page, cfg, before)
    own_host = urlparse(cfg["url"]).netloc.replace("www.", "")
    inside = turn.evaluate("el => [...el.querySelectorAll('a[href^=http]')].map(a => a.href)")
    anywhere = page.evaluate("() => [...document.querySelectorAll('a[href^=http]')].map(a => a.href)")
    uniq = lambda xs: list(dict.fromkeys(clean_url(x) for x in xs))
    
    sources = flatten_sources(extract_payloads(turn, "[data-assistant-sources-payload]", "data-assistant-sources-payload")) \
        if cfg.get("sources_payload") else []
        
    if cfg.get("sources_panel"):
        panel_loc = page.locator(cfg["sources_panel"])
        if panel_loc.count() > 0:
            panel_links = panel_loc.last.evaluate("el => [...el.querySelectorAll('a[href^=http]')].map(a => a.href)")
            sources.extend(panel_links)

    products = extract_payloads(turn, "[data-assistant-product-payload]", "data-assistant-product-payload") \
        if cfg.get("product_payload") else []
        
    return dict(text=prose_text(turn, cfg), full_text=turn_text(turn), html=turn.inner_html(),
                cited_urls=list(dict.fromkeys(clean_url(u) for u in sources + uniq(inside))),
                products=products,
                page_links=[u for u in uniq(anywhere) if own_host not in urlparse(u).netloc])

def find_input_box(page, cfg):
    for sel in cfg["input_box"]:
        loc = page.locator(sel).first
        try:
            loc.wait_for(state="visible", timeout=15000 if SPEED else 1000)
            return loc
        except Exception:
            continue
    return None


def blocked_by_challenge(page):
    try:
        text = page.inner_text("body", timeout=3000).lower()[:3000]
    except Exception:
        return False
    return any(w in text for w in WALL_WORDS)


def get_input_box(page, cfg, engine):
    for _ in range(3):
        box = None if blocked_by_challenge(page) else find_input_box(page, cfg)
        if box:
            return box
        if not sys.stdin.isatty():
            raise Blocked(f"{engine}: captcha/login wall, or input_box selector not found")
        input(f"\n[{engine}] Challenge or missing text box. Fix it in the browser window, then press Enter... ")
    raise Blocked(f"{engine}: still blocked")


def ask_chatbot(browser, engine, cfg, turns, web_search):
    private = cfg.get("private_window", True)
    ctx = browser.new_context(viewport={"width": 1440, "height": 900}) if private else browser.contexts[0]
    page = ctx.new_page()
    try:
        page.goto(cfg["url"], wait_until="domcontentloaded", timeout=60000)
        nap(2, 5)
        if web_search == "off":
            for sel in cfg["web_search_off"]:
                human_click(page, page.locator(sel).first)
                nap(0.8, 2)
        results = []
        for i, t in enumerate(turns):
            if i:
                nap(3, 8)                                                # pause as if reading the first answer
            results.append(send_message(page, cfg, engine, t))
        
        model_shown = None
        if cfg.get("model_label"):
            model_loc = page.locator(cfg["model_label"]).first
            if model_loc.is_visible():
                model_shown = model_loc.inner_text().strip()
        return results, page.evaluate("navigator.userAgent"), model_shown
    except Exception:
        try:
            page.screenshot(path=str(ROOT / f"last_error_{engine}.png"))
        except Exception:
            pass
        raise
    finally:
        page.close()
        if private:
            ctx.close()


# =============================================================== commands
def cmd_check(a):
    ev = load_evaluation(a.evaluation)
    errs, warns = validate(ev)
    [print("WARN :", w) for w in warns]
    [print("ERROR:", e) for e in errs]
    if errs:
        sys.exit(1)
    print("OK:", len(list(all_tests(ev))), "tests,", sum(len(expand_test(t, ev)) for _, t in all_tests(ev)), "distinct prompts")


def cmd_plan(a):
    cmd_check(a)
    ev = load_evaluation(a.evaluation)
    folder, plan_p, answers_p, _ = result_paths(ev, a.wave)
    plan = build_plan(ev, a.core_only)
    text = plan.to_csv(index=False)
    if plan_p.exists() and plan_p.read_text(encoding="utf-8") != text and answers_p.exists() and not a.force:
        die("answers already collected under a different plan. Use a new --wave (or --force).")
    folder.mkdir(parents=True, exist_ok=True)
    plan_p.write_text(text, encoding="utf-8")
    print(f"{len(plan)} runs -> {plan_p}   (~{plan.n_turns.sum() * 1.5 / 60:.1f} h unattended, rough)")
    print(plan.pivot_table(index=["group", "stage"], columns="engine", values="run_id", aggfunc="count", margins=True).fillna(0).astype(int).to_string())
    print("runs per repeat:", plan.groupby("rep").size().to_dict())

PROFILES = {"private": 9222, "regular": 9223}

def cmd_chrome(a):
    cands = [os.environ.get("CHROME_BIN"), shutil.which("google-chrome"), shutil.which("google-chrome-stable"), shutil.which("chromium"),
             "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
             r"C:\Program Files\Google\Chrome\Application\chrome.exe", r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"]
    exe = next((c for c in cands if c and Path(c).exists()), None)
    if not exe:
        die("Chrome not found; set the CHROME_BIN environment variable.")
    regular = a.profile == "regular"
    port = a.port or PROFILES[a.profile]
    udd = Path.home() / ".bench_chrome_regular" if regular else Path(tempfile.gettempdir()) / "bench_chrome"
    cmd = [exe, f"--remote-debugging-port={port}", f"--user-data-dir={udd}", "--no-first-run", "--no-default-browser-check"]
    if not regular:
        cmd.append("--incognito")

    print("launching:", " ".join(cmd))
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def connect(pw, port):
    try:
        return pw.chromium.connect_over_cdp(f"http://localhost:{port}")
    except Exception:
        die(f"can't reach Chrome on port {port}. Run `python bench.py chrome` first.")

def get_browser(pw, cache, cfg, cli_port=None):
    port = cli_port or cfg.get("port", 9222)
    if port not in cache:
        cache[port] = connect(pw, port)
    return cache[port]


def cmd_probe(a):
    from playwright.sync_api import sync_playwright
    cfg = yaml.safe_load(open(ROOT / "engines.yaml"))[a.engine]
    with sync_playwright() as p:
        browser = connect(p, a.port or cfg.get("port", 9222))
        private = cfg.get("private_window", True)
        ctx = browser.new_context(viewport={"width": 1440, "height": 900}) if private else browser.contexts[0]
        page = ctx.new_page()
        page.goto(cfg["url"], wait_until="domcontentloaded"); nap(3, 5)
        print("challenge page detected:", blocked_by_challenge(page))
        for kind in ("input_box", "turn_blocks", "busy_indicator"):
            for sel in cfg.get(kind, []):
                print(f"{kind:15s} {sel:55s} matches={page.locator(sel).count()}")
        if a.ask:
            for i, msg in enumerate(a.ask):
                if i > 0:
                    print(f"\n--- Pausing before turn {i + 1} ---")
                    nap(3, 6)
                print(f"\n[Turn {i + 1}] Asking: {msg}")
                r = send_message(page, cfg, a.engine, msg)
                print(f"[Turn {i + 1}] Answer ({len(r['text'])} chars):\n", r["text"])
                print("Cited:", r["cited_urls"])
                print("Products:", [p.get("title") if isinstance(p, dict) else p for p in r.get("products", [])])

        # html = page.evaluate("""() => {
        #   const out = [];
        #   document.querySelectorAll('main *').forEach(el => {
        #     const attrs = [...el.attributes].filter(a => /^(data-|role|aria-)/.test(a.name))
        #                     .map(a => `${a.name}="${a.value.slice(0,40)}"`).join(' ');
        #     if (attrs) out.push(el.tagName.toLowerCase() + ' ' + attrs);
        #   });
        #   return [...new Set(out)].join('\\n');
        # }""")
        # Path("dom_dump.txt").write_text(html, encoding="utf-8")
        # print("wrote dom_dump.txt")
        input("Press Enter to close... ")
        page.close()
        if private:
            ctx.close()


def cmd_status(a):
    ev = load_evaluation(a.evaluation)
    _, plan_p, answers_p, errors_p = result_paths(ev, a.wave)
    plan = pd.read_csv(plan_p)
    plan["done"] = plan.run_id.isin(read_done(answers_p))
    print(f"{plan.done.sum()}/{len(plan)} runs saved")
    print(plan.groupby(["rep", "engine"]).done.agg(["sum", "count"]).to_string())
    if errors_p.exists():
        print("failed attempts logged:", sum(1 for _ in open(errors_p)))


def cmd_run(a):
    global SPEED
    from playwright.sync_api import sync_playwright
    SPEED = 0.0 if a.fast else 1.0
    ev = load_evaluation(a.evaluation)
    folder, plan_p, answers_p, errors_p = result_paths(ev, a.wave)
    if not plan_p.exists():
        die("no plan yet. Run `python bench.py plan <evaluation>` first.")
    plan = pd.read_csv(plan_p)

    if a.only_test:
        plan = plan[plan.test == a.only_test]
        if plan.empty:
            die(f"no runs for test '{a.only_test}' in {plan_p}")

    todo = plan[~plan.run_id.isin(read_done(answers_p))]
    todo = todo.head(a.limit) if a.limit else todo
    print(f"{len(todo)} runs to do")
    failures_in_a_row = done = 0
    with sync_playwright() as p:
        cache = {}
        # Preflight: fail early if any needed Chrome instance isn't reachable
        for eng in todo.engine.unique():
            get_browser(p, cache, ev["engines"][eng], a.port)
        for _, r in todo.iterrows():
            t0, turns = time.time(), json.loads(r.turns)
            print(f"[{done + 1}/{len(todo)}] {r.engine:10s} {r.prompt_id} rep{r.rep} search={r.web_search}", flush=True)
            try:
                results, ua, model_shown = ask_chatbot(
                    get_browser(p, cache, ev["engines"][r.engine], a.port),
                    r.engine, ev["engines"][r.engine], turns, r.web_search)

                # --- NEW: per-turn detail + whether the final turn produced a fresh answer ---
                turn_results = [
                    {
                        "user": turns[i],
                        "answer": res["text"],
                        "html": res["html"],
                        "cited_urls": res["cited_urls"],
                    }
                    for i, res in enumerate(results)
                ]
                final_turn_is_new = (
                    results[-1]["text"] != results[-2]["text"]
                ) if len(results) > 1 else True

                rec = dict(
                    run_id=r.run_id, prompt_id=r.prompt_id, test=r.test,
                    stage=r.stage, group=r.group, engine=r.engine,
                    web_search=r.web_search, rep=int(r.rep),
                    timestamp=datetime.now().isoformat(timespec="seconds"),
                    turn_results=turn_results,
                    final_answer=results[-1]["text"],
                    n_turns=len(turns),
                    final_turn_is_new=final_turn_is_new,
                    model_shown=model_shown,
                    location_note=ev["profile"].get("location_note"),
                    user_agent=ua,
                    seconds=round(time.time() - t0, 1),
                )
                answers_p.parent.mkdir(parents=True, exist_ok=True)
                with open(answers_p, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                failures_in_a_row = 0
            except KeyboardInterrupt:
                sys.exit("\nstopped; rerun the same command to resume")
            except Exception as e:
                failures_in_a_row += 1
                print("   FAILED:", repr(e)[:200])
                with open(errors_p, "a", encoding="utf-8") as f:
                    f.write(json.dumps(dict(
                        run_id=r.run_id,
                        error=repr(e)[:500],
                        traceback=traceback.format_exc()[-800:],
                    )) + "\n")
                if failures_in_a_row >= 5:
                    die("5 failures in a row: selectors in engines.yaml probably broke. Use `probe`, fix them, rerun.")
            done += 1
            nap(20, 60)                                          # random gap between runs
            if done % random.randint(12, 18) == 0:
                print("   longer break..."); nap(120, 300)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in [("check", cmd_check), ("plan", cmd_plan), ("chrome", cmd_chrome),
                     ("probe", cmd_probe), ("run", cmd_run), ("status", cmd_status)]:
        s = sub.add_parser(name); s.set_defaults(fn=fn)
        if name == "probe":
            s.add_argument("engine")
            s.add_argument("--ask", nargs="+", help="send one or more messages sequentially to test multi-turn chats")
        elif name != "chrome":
            s.add_argument("evaluation")
        s.add_argument("--wave", default=wave_default(), help="results subfolder, default = current quarter")
        s.add_argument("--port", type=int, default=None)          # default None so engines.yaml can supply it
        if name == "chrome":
            s.add_argument("--profile", choices=list(PROFILES), default="private")
        if name == "plan":
            s.add_argument("--core-only", action="store_true", help="leave out diagnostic tests")
            s.add_argument("--force", action="store_true")
        if name == "run":
            s.add_argument("--limit", type=int, help="only do the next N runs (use ~12 as a pilot)")
            s.add_argument("--fast", action="store_true", help="no human pacing (tests only)")
            s.add_argument("--only-test", help="run only a specific test name")
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()