# Calibration and conformal prediction for medical question answering in Arabic and English

Code and summary results for the paper "Calibration and Conformal Prediction for Medical Question Answering in Arabic and English: A Cross-Lingual Study of LLM Uncertainty".

The study asks how well the confidence of two 7B language models matches their accuracy on medical multiple-choice questions in English and Arabic, and whether conformal prediction gives usable coverage guarantees in both languages.

| | English | Arabic |
|---|---|---|
| Dataset | MedMCQA (validation split, single-choice questions only) | MedAraBench (test split) |
| Questions used in the full runs | 2,816 | 4,927 |

| Model | Hugging Face id |
|---|---|
| Qwen2.5-7B | `Qwen/Qwen2.5-7B-Instruct` |
| Jais-adapted-7B | `inception42/jais-adapted-7b-chat` |

Both models are loaded in 4-bit (NF4) and scored with one forward pass per question: the probability of each option letter is read from the next-token distribution, and no text is generated. The prompt style used for all reported runs is `no_prefill`. The random seed is 42.

## Repository contents

| File | What it does |
|---|---|
| `notebooks/colab_setup.ipynb` | Setup notebook: checks the GPU, sets the random seed (42), defines the `set_seed` and `log_run` helpers (runs are logged to `RUN_LOG.md`), and previews the MedAraBench CSV. |
| `notebooks/score_qwen_medmcqa_ar.ipynb` | Scores Qwen2.5-7B. Despite the `_ar` in the file name, the same notebook ran both languages: only `LANG` and `MODEL_NAME` change in the CONFIG cell. |
| `notebooks/score_jais.ipynb` | Scores Jais-adapted-7B, in English and Arabic. |
| `notebooks/analyze_all.ipynb` | Computes accuracy, ECE and Brier score with 95% bootstrap confidence intervals, plus the conformal summary. Writes `summary_metrics.csv` and `summary_conformal.csv`. **All numbers in the paper come from this notebook.** |
| `notebooks/rq3_rq4_conformal_selective_en.ipynb`, `notebooks/rq3_rq4_conformal_selective_ar.ipynb` | Conformal prediction sets, per-subject coverage and risk-coverage curves, for English and Arabic. These notebooks fit the temperature on 25% of the data, so their values can differ slightly from `analyze_all`. |
| `results/summary_metrics.csv`, `results/summary_conformal.csv` | The summary tables written by `analyze_all`. |

## How to reproduce

The notebooks were written for Google Colab with a GPU runtime and Google Drive.

1. Open `notebooks/colab_setup.ipynb` and run it first.
2. In the scoring notebook, edit only the CONFIG cell:
   ```python
   LANG = "en"                                   # "en" = English (MedMCQA), "ar" = Arabic (MedAraBench)
   MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"       # or "inception42/jais-adapted-7b-chat" in score_jais.ipynb
   MEDARA_CSV = "<path to your MedAraBench test CSV>"   # only used when LANG = "ar"
   ```
   Run it once per language and model (four runs in total).
3. If Hugging Face asks for a token, add it to Colab Secrets as `HF_TOKEN`. Never paste a token into a notebook cell.
4. Run `notebooks/analyze_all.ipynb`, then the two `rq3_rq4_conformal_selective_*` notebooks.

The notebooks use paths such as `/content/drive/MyDrive/medqa_results/results/`. Change them to your own folders: `MEDARA_CSV` and the output folder in the scoring notebooks, `BASE` in `analyze_all`, and `RESULTS_FILE` in the RQ3/RQ4 notebooks.

## Result files

Each scoring run writes one `.jsonl` file, named `{model}_{language}_{prompt_style}_seed{seed}_n{N}.jsonl` (for example `Qwen2.5-7B-Instruct_ar_no_prefill_seed42_n4927.jsonl`). One line holds one question: its id, the number of options `K`, the correct option, the predicted option, the confidence, the probability mass on the valid letters, the option probabilities and scores, the subject and the level. No question text is stored in these files. Files with `n50` in the name are small development runs and are not used in the paper. The Jais file names also contain `union`, which marks the scoring that counts both the Latin and the Arabic spelling of each option letter.

## Data

The datasets are not redistributed here.

- MedMCQA is loaded from the Hugging Face hub (`openlifescienceai/medmcqa`).
- MedAraBench (Abu-Daoud et al., ICLR 2026, arXiv:2602.01714): get the test split from the authors' release (see the paper) and follow its license terms. The notebooks expect it as a CSV named `medarabench-test_split.csv` with the columns `Question Number`, `Question`, `Option A` to `Option F`, `Correct Answer`, `Level`, `Medical Specialty`, `Group` and `umbrella_specialty`. Questions have four, five or (rarely) six options.

## Citation

If you use this code, please cite it with the information in `CITATION.cff`.

## License

The code in this repository is released under the MIT License (see `LICENSE`). The datasets keep their own licenses and terms.
