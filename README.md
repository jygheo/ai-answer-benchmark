# Good Culture in AI Answer Engines

This project asks ChatGPT, Gemini and Perplexity the same set of questions and records which brands each one names, which brand it tells the shopper to buy, how it talks about each brand, and which websites it cites. The questions are defined in a YAML file, the answers are collected through a Chrome browser, and the results are scored and charted in a notebook.

## Methodology

**Sample.** Three AI providers were tested: ChatGPT (default), Gemini 3.5 Flash-Lite, and Perplexity (default). Prompts covered four groups: open-category questions (e.g., "best cottage cheese brand?"), feature-led searches (e.g., "highest protein cottage cheese"), shopper-segment searches (e.g., "GLP-1 + protein needs"), and head-to-head comparisons (e.g., Good Culture vs. Daisy).

**Scoring.** Each response was checked for whether a brand was mentioned and whether it was the single top pick. A brand counts as mentioned if the Gemini judge listed it or the text matcher found its name. Top picks were judged by Gemini 3.8 Flash. Results were averaged first by question, then by section.

**Uncertainty.** 95% confidence intervals were calculated using 2,000 rounds of resampling across questions and answer/run observations. Intervals are expected to be wide given the small number of distinct questions per section (5-9).

## Files

| File | What it is |
|---|---|
| `bench.py` | Sends the prompts to the chatbots through Chrome and saves the answers |
| `analyze.py` | Cleans the answers, finds brands and citations, and scores them |
| `analysis.ipynb` | Computes the statistics and draws the charts |
| `engines.yaml` | Web addresses and page selectors for each chatbot (shared by all evaluations) |
| `evaluations/good_culture/brand_profile.yaml` | The brand being measured, the competitor list with alternate spellings, and website categories |
| `evaluations/good_culture/prompts.yaml` | The prompts, which chatbots get them, and how many times |
| `evaluations/good_culture/learned.yaml` | Created by `analyze.py review`. Brands and domains you approved by hand |
| `evaluations/_template/` | Blank, commented copies of the two YAML files above, for starting a new evaluation |
| `evaluations/good_culture/results/<wave>/` | Everything one data collection produces (see [Output files](#output-files)) |

## Prompts

Every prompt is sent to all three chatbots. Reps are repeats of the same prompt in a fresh chat. Placeholders such as `{category}` are filled in from `brand_profile.yaml` before sending.

| Group (stage in YAML) | Distinct prompts | Reps | Runs |
|---|---|---|---|
| Open-category questions (`discovery`) | 5 | 2 | 30 |
| Feature-led searches (`attributes`) | 9 | 2 | 54 |
| Shopper-segment searches (`personas`) | 5 | 2 | 30 |
| Head-to-head comparisons (`comparisons`) | 8 | 1 | 24 |
| **Total** | **27** | | **138** |

That is 46 runs per chatbot. The counts above are for the core tests only (`plan --core-only`) and excluding doubts/switching prompts (unused).

### Open-category questions (5 prompts)

1. `what is the best cottage cheese brand to buy?`
2. `what are some popular new cottage cheese brands?`
3. `what's the best cottage cheese for eating more protein?`
4. `should i eat cottage cheese or greek yogurt for high protein?`
5. Two messages in one chat: `What's the health benefit of cottage cheese? Should I have it?` followed by `Then what brand of cottage cheese should I get?`

### Feature-led searches (9 prompts)

1. `highest protein cottage cheese`
2. `cleanest ingredients cottage cheese`
3. `cottage cheese with live cultures`
4. `best tasting cottage cheese`
5. `lowest sodium cottage cheese`
6. `lactose free cottage cheese`
7. `most affordable cottage cheese`
8. `easiest cottage cheese to find in stores`
9. `can you recommend a cottage cheese with live cultures, no gums or thickeners, and lots of protein?`

### Shopper-segment searches (5 prompts)

Each is two messages in one chat. The first describes the shopper. The second is always `Which brand of cottage cheese should I get?` Only the answer to the second message is scored.

1. GLP-1 user: `I'm on a GLP-1 and trying to get enough protein while I'm losing weight.`
2. Toddler parent: `I'm looking for some healthy dairy snacks for my toddler. I'd rather avoid a lot of added stuff.`
3. Lactose intolerant: `I like dairy but regular dairy doesn't always agree with my stomach.`
4. Budget shopper: `I'm trying to eat more protein without spending a ton at the grocery store.`
5. Low sodium: `I've been told to cut back on sodium, so I'm trying to be more careful about what I buy.`

### Head-to-head comparisons (8 prompts)

The template is `{target} vs {rival}, which should I buy?` with four rivals: Daisy, Breakstone's, Knudsen and store brand. Each is asked in both orders, so there are 8 prompts. For example:

- `Good Culture vs Daisy, which should I buy?`
- `Daisy vs Good Culture, which should I buy?`

Asking both orders keeps the result from depending on which brand is named first. Each prompt is run once per chatbot.

## Running a wave

### 1. Install

```
pip install -r requirements.txt
```

You also need Google Chrome. The scripts connect to a Chrome you start yourself.

### 2. Collect answers with `bench.py`

Run these in order from the repo root.

```
python bench.py check good_culture
python bench.py plan good_culture --core-only --wave 2026_Q4
python bench.py chrome
python bench.py chrome --profile regular
python bench.py probe chatgpt
python bench.py run good_culture --wave 2026_Q4 --limit 12
python bench.py run good_culture --wave 2026_Q4
python bench.py status good_culture --wave 2026_Q4
```

### 3. Process the answers with `analyze.py`

```
python analyze.py check good_culture --wave 2026_Q4
python analyze.py parse good_culture --wave 2026_Q4
python analyze.py review good_culture --wave 2026_Q4
python analyze.py parse good_culture --wave 2026_Q4
python analyze.py judge-api good_culture --wave 2026_Q4
```

### 4. Run the notebook

```
GC_EVAL_DIR=evaluations/good_culture GC_WAVE=2026_Q4 jupyter notebook analysis_updated.ipynb
```

The notebook reads the CSVs from `results/<wave>/tables/` and writes `scoreboard.csv`, other summary tables, and chart images.

## What each script does

**`bench.py`** automates asking the chatbots. It turns `prompts.yaml` into a list of runs, opens each chatbot in Chrome through Playwright, types the prompt, waits for the answer to finish, and saves the text, the HTML and the cited URLs for every turn. It also records the model name shown on the page and the location note. It keeps going through failures, logs them to `errors.jsonl`, and can be stopped and resumed.

**`analyze.py`** turns raw answers into tables. It filters bad runs, finds brands and citations, and has Gemini score each answer. It does not compute any statistics or charts. Its output is a set of CSVs the notebook reads.

**`analysis_updated.ipynb`** does the statistics and charts: mention rates, top-pick rates, head-to-head results, heatmaps, tone, cross-engine comparison and citation mix. It ends with a summary table of findings.

**`engines.yaml`** tells `bench.py` how to use each chatbot's page: the address, the CSS selectors for the text box and the answer, how to tell the answer is still being written, and whether the chatbot needs a logged-in profile. When a chatbot changes its page layout, this is the file to edit.

## What goes into each result

The notebook uses the open-category, feature-led and shopper-segment prompts (19 distinct prompts, 114 runs) for most results, and the head-to-head prompts for the head-to-head result.

| Result in the notebook | Prompts used | Distinct prompts | Answers used |
|---|---|---|---|
| Unbranded search visibility (% of answers naming each brand) | Open-category, feature-led, shopper-segment | 5 / 9 / 5 (19 total) | 30 / 54 / 30 (114 total), 38 per chatbot |
| Selection among mentions (top-pick rate ÷ mention rate) | Same as above | 19 | 114 |
| Head-to-head | Head-to-head vs. Daisy, Breakstone's, Knudsen, both orders | 6 of the 8 | 18 (6 per rival) |
| Positioning by product attribute | Feature-led | 9 | 54 (about 6 per prompt) |
| Positioning by shopper segment | Shopper-segment | 5 | 30 (about 6 per prompt) |
| Tone (average sentiment when a brand is named) | Open-category, feature-led, shopper-segment | 19 | 114, but only answers that name the brand |
| Cross-engine consistency | Open-category, feature-led, shopper-segment | 19 | 114, split 38 per chatbot |
| Citation source mix | Open-category, feature-led, shopper-segment | 19 | Cited links in those 114 answers |


## Output files

Everything for a wave is in `evaluations/<name>/results/<wave>/`.

| File or folder | Created by | Contents |
|---|---|---|
| `plan.csv` | `bench.py plan` | Every run to be done |
| `answers.jsonl` | `bench.py run` | Every saved answer, one line per run |
| `errors.jsonl` | `bench.py run` | Failed attempts |
| `tables/quality_report.csv` | `analyze.py check` | Which runs were excluded and why |
| `tables/mentions.csv` | `analyze.py parse` | Brands found in each answer, with list position |
| `tables/citations.csv` | `analyze.py parse` | Each cited link, its website, and its type |
| `tables/unknown_brands.csv`, `unknown_domains.csv` | `analyze.py parse` | Candidates for `review` |
| `tables/judgments.csv` | `analyze.py judge-*` | Gemini's scores, one row per brand per answer |
| `judge/` | `analyze.py judge-*` | Saved progress, rejected items, and which model judged |
| `tables/scoreboard.csv` and other summaries | notebook | Final numbers with ranges |
| `figures/` | notebook | Chart images |

## Starting a new evaluation

A new brand, category or question set gets its own folder. `evaluations/_template/` contains blank copies of `brand_profile.yaml` and `prompts.yaml` with comments explaining each field.

```
cp -r evaluations/_template evaluations/my_brand
```

1. Edit `evaluations/my_brand/brand_profile.yaml`. Set the category, the target brand, the market leader, and the list of brands with any alternate spellings. A brand's website goes in its alias list (an alias containing a dot is treated as that brand's own domain). Fill in the `domains` section for retailers, editorial sites, health sites and community sites.
2. Edit `evaluations/my_brand/prompts.yaml`. Choose the chatbots, then define each test. Every test needs a `name`, a `stage`, and `turns`. The stage must be one of `discovery`, `attributes`, `personas`, `comparisons`. Use `matrix` to produce several prompts from one test, `reps` to repeat each, and `counterbalance: true` with a `rival` key to ask in both orders.
3. Available placeholders in prompt text: `{category}`, `{target}`, `{market_leader}`, and the name of any key in the test's `matrix`.
4. Run `python bench.py check my_brand`. It reports unknown placeholders, duplicate test names and brands missing from the profile.
5. Collect and process as described in [Running a wave](#running-a-wave), replacing `good_culture` with `my_brand`.

To repeat the evaluation for the same brand later, keep the same folder and use a new wave name such as `2027_Q1`.

