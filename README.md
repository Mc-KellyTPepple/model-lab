# Model Lab

Upload a spreadsheet, pick the column to predict, and get a model plus an honest account of how far to trust it.

Most "AutoML" demos show one impressive number. Model Lab is built around the checks that stop that number from lying:

1. **Profile.** Types, missing values, duplicates, ID-like and free-text columns that should not be used.
2. **Leak scan.** A column that alone nearly gives the answer (for example `account_status` when predicting churn) is left out, and the score *with* it is shown for contrast.
3. **Fair comparison.** Five models (a baseline that always guesses the most common answer, a small tree, linear/logistic, gradient boosting, random forest). 20% of the rows are set aside first. The rest is scored with 5-fold cross-validation, with all preprocessing fitted inside each fold.
4. **Choosing.** The simplest model within one standard error of the best. A paired comparison against the baseline decides whether the model is called useful at all.
5. **One look at the test rows**, with a bootstrap 95% interval, and the baseline's score next to it.
6. **Explain.** Permutation importance, a confusion matrix or predicted-vs-actual plot.
7. **Deploy.** A prediction form generated from the data; it warns when a value is outside the range the model saw.
8. **Monitor.** Upload newer data: per-column drift (Population Stability Index with a sampling-noise floor), the prediction mix, and the real score if the answers are included.

Scores use balanced accuracy when one category is 70% or more of the rows (plain accuracy would reward always guessing it) and R² for numbers.

## Limits, stated plainly

* Tables only (CSV, up to 1 MB, 5,000 rows, 40 columns). Large files are sampled to 2,000 rows to keep the demo quick.
* Five classical models with sensible defaults, no hyperparameter search. It is a careful first pass, not a competition entry.
* The leak scan catches a single column that nearly solves the problem. It cannot catch a leak spread across several columns, or one that is not statistical (a column that is only available after the outcome but weakly related).
* Uploaded files stay in memory and are forgotten after an hour. Nothing is stored or logged.

## Run it

```
pip install -e .
python -m modellab          # serves on :7860
```

```
GET  /samples, /samples/<id>.csv      practice datasets (three real, three synthetic and labelled so)
POST /profile                         CSV -> columns, warnings, suggested target
POST /train?target=&task=&leaks=keep  CSV or {"sample": id} -> {"job": id}
GET  /jobs/<id>                       progress, then the full result
POST /predict                         {"session": id, "values": {...}}
POST /monitor?session=id              CSV or {"sample": id} -> drift report
```

## Benchmark

`python -m modellab.bench` (GitHub Actions: *Benchmark on public datasets*) holds back 30% of each public dataset's rows where Model Lab cannot see them, and checks its claims against them: pick quality, how close its stated score is to the kept-back score, false alarms on shuffled answers, and planted-leak detection. Results are committed to `evals/results/`.

Results: see the table below once the first run has finished.

## Tests

```
pip install pytest && python -m pytest
```

Includes the checks that matter most: a target of pure noise must not be called useful, a planted leak must be caught and excluded, and small data must carry a warning.
