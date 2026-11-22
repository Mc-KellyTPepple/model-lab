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

See Results below.

## Results

11 public datasets, 3 different splits each (33 runs). Two more (spambase, qsar-biodeg) were rejected for having more than the 40-column limit, so they are not counted. Everything below is read from `evals/results/` and was produced by the Action, not typed in.

| claim | result |
|---|---|
| Shuffled answers wrongly called useful | 2 of 99 |
| Planted answer-giveaway column caught | 33 of 33 (false flags on unmodified data: 0 of 33) |
| Chosen model within 2 points of the best candidate on kept-back rows | 82% of runs (median regret 0.2 points) |
| Typical gap between stated test score and kept-back score | 1.5 points; the 95% interval contained it in 88% of runs |
| Called the model useful, and it really beat the baseline | 31 of 31 |

By dataset (averages over the splits; numbers for classification are accuracy or balanced accuracy, regression is R²):

| dataset | rows | stated | kept-back | baseline | leak caught | false alarms |
|---|---|---|---|---|---|---|
| credit-g | 1,000 | 71.8% | 71.9% | 64.3% | 3/3 | 0/9 |
| diabetes-pima | 768 | 75.9% | 76.5% | 62.8% | 3/3 | 0/9 |
| blood-transfusion | 748 | 62.6% | 63.7% | 50.0% | 3/3 | 1/9 |
| phoneme | 5,404 | 85.4% | 84.3% | 50.0% | 3/3 | 0/9 |
| kc1 | 2,109 | 61.6% | 63.7% | 50.0% | 3/3 | 0/9 |
| ilpd | 583 | 57.1% | 60.2% | 50.0% | 3/3 | 0/9 |
| banknote | 1,372 | 98.4% | 98.4% | 56.0% | 3/3 | 0/9 |
| vehicle | 846 | 78.4% | 78.3% | 23.0% | 3/3 | 1/9 |
| segment | 2,310 | 99.0% | 98.1% | 12.7% | 3/3 | 0/9 |
| abalone | 4,177 | 0.51 | 0.51 | -0.00 | 3/3 | 0/9 |
| california-housing | 20,640 | 0.74 | 0.78 | -0.00 | 3/3 | 0/9 |

Reading it honestly:

* The kept-back rows are themselves a sample, so an interval that is exactly right would still miss them somewhat more than 5% of the time. Coverage of about 88% is in that range; the typical gap is the number to read.
* In two runs (credit-g and diabetes-pima, one split each) it said "not clearly better than the baseline" when the model did beat it on the kept-back rows. That is the cautious direction to be wrong in.
* Two of the 99 shuffled-answer runs (blood-transfusion and vehicle, one each) were wrongly called useful. With 99 runs that is expected to happen occasionally for any test at this strictness.
* The metric switches between accuracy and balanced accuracy when the most common answer crosses 70% of the training rows, so a dataset sitting right on that line (credit-g) can use a different metric on different splits.

### What changed after the first run

The first run loaded only 5 of 13 datasets (a loader bug in the benchmark script, now fixed), and the leak scan caught the planted leak in 12 of 15 runs. All three misses were the 7-class `segment` dataset: a shallow decision tree on one-hot columns cannot isolate that many classes, and the same weakness would hit a noisy numeric copy of the answer. The scan now looks up the most common answer per category and uses 24-leaf trees for numbers, with tests for both cases. The second run is the one reported above; the first run's numbers are in the Git history.

### Timing on the live server

Training each built-in sample on the deployed service, from a GitHub-hosted runner: 10 to 26 seconds (`evals/results/live.json`).

## Tests

```
pip install pytest && python -m pytest
```

Includes the checks that matter most: a target of pure noise must not be called useful, a planted leak must be caught and excluded, and small data must carry a warning.
