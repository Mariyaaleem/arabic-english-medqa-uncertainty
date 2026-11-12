# Calibration and conformal prediction for medical question answering in Arabic and English

Code and summary results for the paper "Calibration and Conformal Prediction for Medical Question Answering in Arabic and English: Evidence from Two Non-Parallel Benchmarks".

The study asks how well the confidence of two 7B language models matches their accuracy on medical multiple-choice questions in English and Arabic, and whether conformal prediction gives usable coverage guarantees in both languages. The two benchmarks are not parallel translations, so differences between languages are not attributed to language itself.

| | English | Arabic |
|---|---|---|
| Dataset | MedMCQA (validation split, single-choice questions only) | MedAraBench (test split) |
| Questions used in the full runs | 2,816 | 4,927 |

| Model | Hugging Face id |
|---|---|
| Qwen2.5-7B | `Qwen/Qwen2.5-7B-Instruct` |
| Jais-adapted-7B | `inception42/jais-adapted-7b-chat` (revision `f2de64b0`) |

Both models are loaded in 4-bit (NF4, float16 compute) and scored with one forward pass per question: the probability of each option letter is read from the next-token distribution, and no text is generated. The prompt style used for all reported runs is `no_prefill`. The random seed is 42.

## Repository contents

| File | What it does |
|---|---|
| `notebooks/colab_setup.ipynb` | Setup notebook: checks the GPU, sets the random seed (42), defines the `set_seed` and `log_run` helpers (runs are logged to `RUN_LOG.md`), and previews the MedAraBench CSV. |
| `notebooks/score_qwen_medmcqa_ar.ipynb` | Scores Qwen2.5-7B. Despite the `_ar` in the file name, the same notebook ran both languages: only `LANG` and `MODEL_NAME` change in the CONFIG cell. |
| `notebooks/score_jais.ipynb` | Scores Jais-adapted-7B, in English and Arabic. Letter scoring is controlled by `USE_ARABIC_LETTERS` in the prompt cell. |
| `notebooks/analyze_all.ipynb` | Computes accuracy, ECE (raw and temperature scaled), Brier score, AUROC and risk-coverage of the top-two margin, with 95% bootstrap confidence intervals, plus the conformal summary and the answer-position tables. Writes `summary_metrics.csv` and `summary_conformal.csv`. Tables 2 to 6 and Appendix A of the paper come from this notebook. |
| `notebooks/rq3_rq4_conformal_selective_en.ipynb`, `notebooks/rq3_rq4_conformal_selective_ar.ipynb` | Conformal prediction sets, per-subject coverage and risk-coverage curves, for English and Arabic. These notebooks fit the temperature on 25% of the data, so their values can differ slightly from `analyze_all`. |
| `appendix_c_checks.py` | Robustness checks of Appendix C from the saved `.jsonl` files: reproduction check of Tables 2 and 3, Latin-only versus union scoring (Table C1), bootstrap intervals for conformal coverage and set size (Table C2), and AUROC and AURC of the margin versus the maximum probability (Table C3). |
| `results/summary_metrics.csv`, `results/summary_conformal.csv` | The summary tables written by `analyze_all`. |

## How to reproduce

### 1. Environment

All runs used Google Colab with one NVIDIA Tesla T4 (about 15 GB) and fixed seed 42.

| Component | Version |
|---|---|
| Python | 3.13 |
| PyTorch | 2.11 |
| Transformers | 5.22 |
| Accelerate | 1.15 |
| bitsandbytes | 0.50.2 |
| scikit-learn / SciPy / NumPy / pandas | 1.6 / 1.16 / 2.4 / 2.2 |

Results depend on library versions and on 4-bit quantisation, so use these versions. The notebooks were written for Google Colab with a GPU runtime and Google Drive.

### 2. Step 1: option scores

1. Open `notebooks/colab_setup.ipynb` and run it first.
2. In the scoring notebook, edit only the CONFIG cell:

   ```python
   LANG = "en"                                   # "en" = English (MedMCQA), "ar" = Arabic (MedAraBench)
   MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"       # or "inception42/jais-adapted-7b-chat" in score_jais.ipynb
   MEDARA_CSV = "<path to your MedAraBench test CSV>"   # only used when LANG = "ar"
   ```

3. Run the setup, data, model and prompt cells, then the evaluation loop. It resumes automatically after a Colab disconnect. Run it once per language and model (four runs in total).
4. If Hugging Face asks for a token, add it to Colab Secrets as `HF_TOKEN`. Never paste a token into a notebook cell.

The notebooks use paths such as `/content/drive/MyDrive/medqa_results/results/`. Change them to your own folders: `MEDARA_CSV` and the output folder in the scoring notebooks, `BASE` in `analyze_all`, and `RESULTS_FILE` in the RQ3/RQ4 notebooks.

**Letter scoring for Jais** is controlled in the prompt cell of `score_jais.ipynb` by `USE_ARABIC_LETTERS`:

- `True` (used for all main results, "union" scoring): the probability of each Arabic letter (أ، ب، ج، د، ه، و) is added to that of the Latin letter.
- `False` ("Latin only", Appendix C): Latin letters only.

### 3. Result files

Each scoring run writes one `.jsonl` file, named `{model}_{language}_{prompt_style}_seed{seed}_n{N}.jsonl` (for example `Qwen2.5-7B-Instruct_ar_no_prefill_seed42_n4927.jsonl`). One line holds one question, with these fields: `id`, `K` (number of options), `answer`, `pred`, `conf`, `mass`, `probs`, `scores`, `specialty` and `level`.

- `scores` are the log-probabilities of the options before normalisation, `mass` is the probability the model put on valid option letters, and `probs` is the normalised distribution.
- No question text is stored in these files.
- Files with `n50` in the name are small development runs and are not used in the paper.
- Jais files created with union scoring have `union` in their name (for example `jais-adapted-7b-chat_ar_no_prefill_union_seed42_n4927.jsonl`) and are in `medqa_results/results/`. Latin-only Jais files have no `union` in their name and are in `medqa_results/ablation/`.

### 4. Step 2: main analysis (Tables 2 to 6)

Run `notebooks/analyze_all.ipynb` on the `.jsonl` files, then the two `rq3_rq4_conformal_selective_*` notebooks. The analysis computes accuracy, raw and temperature-scaled ECE (15 equal-width bins), Brier score, AUROC and risk-coverage of the top-two margin, split conformal prediction (25% / 25% / 50% split, 100 random splits) and the answer position tables.

Sanity check: your numbers should match these.

| Setting | Accuracy | Raw ECE | AUROC (margin) | Temperature T |
|---|---|---|---|---|
| Qwen, English | 0.561 | 0.386 | 0.734 | 8.71 |
| Qwen, Arabic | 0.430 | 0.486 | 0.646 | 10.90 |
| Jais, English | 0.369 | 0.242 | 0.600 | 3.40 |
| Jais, Arabic | 0.308 | 0.309 | 0.579 | 5.70 |

### 5. Step 3: Appendix C

Appendix C uses the same `.jsonl` files and a separate script, `appendix_c_checks.py`. Give it the folder that contains `results/` and `ablation/`:

```bash
python appendix_c_checks.py medqa_results            # all parts: reproduction check, C.1, C.2, C.3
python appendix_c_checks.py medqa_results --parts c1 # one part only (repro, c1, c2, c3)
```

The script needs only numpy and scipy and prints the tables in the layout of Tables C1 to C3. Its random splits are not identical to those of the main analysis, so scaled ECE can differ by up to about 0.006, fitted temperatures by up to about 0.25 and mean set sizes by up to about 0.04.

- **C.1 Latin-only vs union scoring.** Run Step 1 for Jais with `USE_ARABIC_LETTERS = False` (English and Arabic) and compare with the union files. Before the full run, two checks should pass: with union scoring switched on, the scores of the first 20 questions must reproduce the saved union file (maximum difference 0.0), and Latin-only scoring on the 50-question Arabic development sample should give a mean mass on letters of about 0.497. The Arabic run takes about 24 minutes on a T4. The paired bootstrap for the differences in accuracy, AUROC and ECE (2,000 resamples over questions) and the exact McNemar test are in `appendix_c_checks.py`.
- **C.2 Intervals for conformal coverage and set size.** 500 bootstrap replicates, each with a random 25/25/50 split and questions resampled within each part, alpha = 0.10.
- **C.3 AUROC and AURC, margin vs maximum probability.** 500 bootstrap resamples over questions for the AUROC intervals. AURC is the mean of the risk when answering the top k questions, over all k.

### 6. Notes

- Reported intervals are bootstrap or split-to-split intervals as described in the paper; they do not cover prompt choice or model choice.
- The study uses exam-style multiple-choice questions. It is not clinical validation and does not support claims about clinical decision support.

## Data

The datasets are not redistributed here.

- MedMCQA is loaded from the Hugging Face hub (`openlifescienceai/medmcqa`) and is released under Apache 2.0 according to its dataset card.
- MedAraBench (Abu-Daoud et al., ICLR 2026, arXiv:2602.01714): get the test split from the authors' release (see the paper) and follow its license terms. Its repository states no license, so its questions are not included here. The notebooks expect it as a CSV named `medarabench-test_split.csv` with the columns `Question Number`, `Question`, `Option A` to `Option F`, `Correct Answer`, `Level`, `Medical Specialty`, `Group` and `umbrella_specialty`. Questions have four, five or (rarely) six options. The released file has 4,959 rows; 32 invalid rows are removed, leaving 4,927 questions.

## Citation

If you use this code, please cite it with the information in `CITATION.cff`.

## License

The code in this repository is released under the MIT License (see `LICENSE`). The datasets keep their own licenses and terms.
