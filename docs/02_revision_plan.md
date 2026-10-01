# JIAP-2026-490298 — Revision plan

Decision: **Major Revision**, due **16 August 2026** (20 days from 27 July).
A separate point-by-point Responses file is required at resubmission.

---

## Part A — The three editor comments

### Comment 1: training data, hyperparameter tuning, validation protocol, public code

The current Methods section says training samples were "collected manually"
with no counts, gives no split, and states the RF used "the default GEE
configuration of 10 trees". That last sentence is the single most damaging line
in the paper: it concedes that no tuning was done, while the whole contribution
is a classifier comparison. An untuned SVM against an untuned RF is not a fair
benchmark, and the editor has noticed.

What to produce:

- A sample-design table — points per class, split into train and test, plus the
  sampling rule (minimum separation distance, source imagery, interpreter).
  `src/train_compare.py` writes `results/tables/sample_design.csv` for this.
- An explicit protocol paragraph: stratified 70/30 split, 5-fold stratified CV
  on the training half for the search, single untouched test half for reported
  accuracy, fixed seed 42.
- A hyperparameter table with the searched ranges and the selected values.
  `results/tables/best_hyperparameters.json` after tuning.
- A public repository. Mirror the cloned GEE repo to GitHub (see
  `docs/01_import_gee_locally.md`, step 3), archive it on Zenodo for a DOI, and
  rewrite the Data Availability Statement. The current wording — "available
  from the corresponding author upon reasonable request" — is exactly what the
  editor is objecting to. `gee_repo/taipei_lulc_reference.js` is a cleaned
  script ready to seed that repo.

### Comment 2: modern classifiers and significance testing

CART, SVM and RF are a 2005-era comparison set. Add XGBoost, LightGBM and a
small MLP; `src/train_compare.py` handles all three. GEE also has
`smileGradientTreeBoost` natively if you want a comparator that stays in the
cloud — it's already wired into the reference script.

For significance, `src/mcnemar_test.py` runs pairwise McNemar with an exact
binomial test when discordant counts are small and Holm–Bonferroni correction
across the pairs.

#### Results, and the uncomfortable finding

All six classifiers have now been tuned and compared on the corrected split
(7,378 train / 1,999 validation pixels, split at polygon level):

| Model | CV acc | OA | OA 95% CI | Kappa | Tuning (s) | Inference (ms) |
|---|---|---|---|---|---|---|
| SVM (RBF) | 0.983 | **0.963** | 0.954–0.971 | 0.944 | 7.5 | 20.6 |
| MLP (deep) | 0.978 | 0.938 | 0.927–0.948 | 0.907 | 93.6 | 1.1 |
| Random Forest | 0.982 | 0.932 | 0.920–0.943 | 0.898 | 58.2 | 12.4 |
| LightGBM | 0.968 | 0.932 | 0.920–0.943 | 0.897 | 1126.6 | 13.2 |
| XGBoost | 0.965 | 0.928 | 0.917–0.939 | 0.892 | 21.1 | 5.3 |
| CART | 0.952 | 0.923 | 0.911–0.935 | 0.885 | 2.2 | 0.3 |

The headline is that the gradient-boosting methods the editor asked for do not
win. SVM does, and CART — the simplest model in the set — is within four points
of it. That is worth saying directly rather than burying: on four Sentinel-2
bands and five spectrally distinct classes, there is not much non-linear
structure left for a boosted ensemble to find.

Then the significance testing, which is where it gets interesting. Run naively
at the pixel level, 9 of the 15 pairwise comparisons come out significant after
Holm correction, several at p < 10⁻¹⁰. Run with the polygon clustering taken
into account — the Durkalski et al. (2003) clustered McNemar statistic, which
sums discordances within polygon before squaring — **none of the 15 survive**.
Not one. The best case is MLP vs CART at Holm-corrected p = 0.51.

| Comparison | ΔOA | p (naive, Holm) | p (clustered, Holm) |
|---|---|---|---|
| SVM vs CART | +0.040 | 2.7 × 10⁻¹⁵ | 1.00 |
| SVM vs XGBoost | +0.035 | 7.8 × 10⁻¹⁴ | 1.00 |
| SVM vs Random Forest | +0.032 | 5.2 × 10⁻¹² | 1.00 |
| MLP vs CART | +0.015 | 1.3 × 10⁻⁵ | 0.51 |
| MLP vs Random Forest | +0.007 | 0.0078 | 0.35 |
| Random Forest vs LightGBM | 0.000 | 0.84 | 1.00 |

The reason is visible in the discordance counts. SVM beats Random Forest on 68
pixels and loses on 5, which looks decisive until you notice those 68 pixels sit
in only 8 polygons — and within a polygon they are essentially the same
observation repeated. Eight informative locations cannot separate a
three-point accuracy difference.

This was validated by simulation before being trusted. Under a null where two
equivalent models both make polygon-clustered errors, the naive test rejects
**44.8%** of the time at α = 0.05; the clustered test rejects 5.1%, which is the
nominal rate. The clustered test still detects real differences when they are
large (power 0.96 for a strong effect), so this is not a test that simply
refuses to reject.

So the honest sentence for the manuscript is: *no classifier was significantly
better than any other once the clustered structure of the validation sample was
accounted for, despite apparent differences of up to 4 percentage points in
overall accuracy.* Report both columns and let the reader see the gap. This is a
stronger contribution than a spurious ranking, and it is the direct answer to
the editor's question — the observed performance differences are **not**
demonstrably meaningful on the present validation design.

It also means the model choice should be made on other grounds. CART trains in
2.2 s and predicts in 0.25 ms; LightGBM takes 1,127 s to tune for no measurable
accuracy gain. That is a defensible basis for a recommendation when accuracy
cannot distinguish the candidates, and it feeds the computational-efficiency
reporting the editor also asked for.

#### The fix

Enlarge the reference set. Stratified random sampling with proportional
allocation, targeting roughly 500–750 points, would give the minority classes
(swamp, water) enough support to be estimated at all. At present swamp has 4–18
reference pixels depending on which table you read, which is why its producer's
accuracy swings between 20% and 37.5%. `src/draw_reference_sample.py` designs
this sample; once it is interpreted, re-run `mcnemar_test.py --no-cluster-correction`
on it, because a probability sample of independent points does not need the
clustering correction and will recover real statistical power.

### How large the validation sample really is

The corrected pipeline produces 1,999 validation pixels, which sounds like
twenty times the 100 points the paper claims. It is not, because those pixels
come from only 30 polygons and pixels inside a polygon are near-duplicates.
Measuring the intra-polygon correlation on the actual downloaded data:

| Class | val pixels | val polygons | ICC | mean cluster | design effect | effective n |
|---|---|---|---|---|---|---|
| Swamp | 135 | 2 | 0.78 | 67.5 | 52.9 | 2.6 |
| Water | 167 | 3 | 0.86 | 55.7 | 47.8 | 3.5 |
| Plantation | 168 | 5 | 0.56 | 33.6 | 19.1 | 8.8 |
| Forest | 1,048 | 13 | 0.84 | 80.6 | 67.5 | 15.5 |
| Built-up | 481 | 7 | 0.29 | 68.7 | 20.9 | 23.0 |
| **Total** | **1,999** | **30** | | | | **≈53** |

Kish's design effect is `deff = 1 + (m − 1)·ICC` and the effective sample size
is `n / deff`. So 1,999 validation pixels carry roughly the information of 53
independent observations — *fewer* than the 100 reference points the paper
already claims. Swamp's producer's accuracy rests on two independent locations.

This settles the sufficiency question in both directions. **Training is
adequate**: 7,378 pixels across 80 polygons is more than enough for these
classifiers, with the caveat that swamp is learned from seven locations, which
bounds how far the model can transfer. **Validation is not adequate**, and no
amount of reweighting fixes it — which is precisely why the editor's request for
significance testing cannot be satisfied honestly without a new reference
sample.

Report the effective sample size in the Limitations subsection. It converts a
vague concession about sample size into a number, and it is a far stronger
answer to the editor than quietly citing 1,999.

### Area-weighted accuracy — Olofsson et al. (2014)

`src/olofsson.py` implements the good-practice stratified estimator from
Olofsson et al. (2014), *Remote Sensing of Environment* 148: 42–57. It fixes a
second, separate problem: the validation sample's class mix is an accident of
where polygons were drawn, not a design. Forest supplies 1,048 of the 1,999
pixels, so the unweighted overall accuracy is largely a statement about forest.

The estimator reweights each map class by the fraction of the map it occupies
(`W_i`), turning the confusion matrix into an estimate over the map rather than
over the sample, and yields area estimates with confidence intervals. Those
intervals are what the revised Results should report instead of bare area
figures. The implementation was validated by Monte Carlo against a known
population: overall-accuracy bias below 0.0001 and formula-based standard errors
within 0.5% of the empirical spread.

Two honest caveats, both encoded in the script and both belonging in the paper:

1. The estimator assumes a probability sample within each stratum. Ours is
   clustered, so `olofsson.py` substitutes the effective sample size above for
   `n_i` in every variance term. Without that correction the standard errors are
   roughly six times too narrow — overall accuracy comes out as ±0.9 percentage
   points instead of ±5.2.
2. The sample is also *purposive* — polygons were placed on unambiguous,
   spectrally pure examples, which systematically excludes the mixed and edge
   pixels where classifiers actually fail. This biases accuracy upward by an
   amount the data cannot reveal. Only a new sample fixes it.

Olofsson et al. also argue against reporting Kappa. Given that the manuscript
currently leads with three Kappa values, dropping them needs a sentence of
justification rather than silent removal.

### Sizing the new reference sample

`src/draw_reference_sample.py` implements Olofsson's sample-size formula
(eq. 13) and draws the points. Using the current accuracy estimates as priors:

| Target SE on OA | Required n | Drawn after per-class floor |
|---|---|---|
| 0.020 (±3.9 pp) | 152 | 377 |
| 0.015 (±2.9 pp) | 270 | 441 |
| 0.010 (±2.0 pp) | 608 | 715 |

The floor matters more than the total. Strict proportional allocation at n=600
would give swamp 38 points and water 38; the script imposes a minimum of 75 per
class, which is why the drawn totals exceed the formula's n. Olofsson explicitly
endorses departing from proportional allocation for rare classes, trading some
precision on overall accuracy for per-class estimates that exist at all.

The 715-point option lands inside the 500–750 range above and gives every class
enough support for McNemar to be meaningful. At roughly 30 seconds per point in
a high-resolution basemap, that is about six hours of interpretation — the
single largest time commitment in the revision, and worth scheduling first.

Points are exported shuffled with the map class hidden from row order. Interpret
them blind: if the interpreter can see what the map said, the reference labels
inherit the map's errors and the assessment is circular.

### Comment 3: deeper discussion

Five things to add, each currently absent:

**Causes of misclassification.** `src/error_analysis.py` now answers this
with numbers rather than the usual appeal to "spectral similarity", and the
answer is not what the confusion matrix suggests.

Jeffries-Matusita separability on the training spectra says the class
definitions are almost all fine. Nine of the ten class pairs score above the
conventional JM = 1.9 threshold. Only one pair is marginal:

| Pair | JM | Verdict |
|---|---|---|
| Built-up ↔ Swamp | 1.851 | marginal |
| Water ↔ Built-up | 1.987 | separable |
| Forest ↔ Plantation | 1.989 | separable |
| Water ↔ Swamp | 1.989 | separable |
| all others | ≥ 1.998 | separable |

So the forest/plantation pair, which the confusion matrix makes look like the
central problem, is spectrally separable. Something else is going on, and the
error concentration table finds it: **97% of every model's forest errors come
from one polygon.** Polygon 77 contributes 92 of Random Forest's 98 forest
errors, 41 of SVM's 42, and the same story for the other four classifiers. The
remaining twelve forest polygons contribute a handful of pixels between them.
The Mahalanobis check ranks polygon 77 as the most spectrally atypical forest
polygon in the dataset, 2.6× the class median distance from the forest centroid.

Swamp is the same shape of problem, and worse because swamp has only two
validation polygons. Polygon 90 supplies 22 of the 23 swamp→water errors under
SVM. Its mean NDWI is +0.17 against +0.33 for true water and −0.25 for the other
swamp polygon; its NIR reflectance is 0.077 against 0.190 for polygon 88. It is
spectrally open water. Whether that is a labelling error or a genuinely
inundated wetland at the July 2020 acquisition is a question only visual
inspection can settle, but either way the reported swamp producer's accuracy
is a statement about one 30-pixel patch.

Built-up→swamp is the third recurring confusion, and it too localises: polygon
16 holds 100% of it for XGBoost, LightGBM and the MLP. That one probably *is*
the marginal JM pair showing up, since built-up↔swamp is the only pair below
threshold — dark, wet, low-albedo urban surfaces against saturated wetland.

The Discussion sentence that follows from this is much stronger than the
generic one: *the residual error is not distributed spectral confusion but a
small number of spatially localised, spectrally atypical training polygons; with
polygon 77 and polygon 90 excluded, the forest/plantation and swamp/water
confusions largely disappear.* Include the separability heatmap
(`results/figures/separability_heatmap.png`) and the error-concentration table,
and add the polygon re-inspection to the Limitations.

One caution before acting on it: do not simply delete the flagged polygons and
re-report the improved accuracy. That is fitting the reference data to the map.
Re-inspect them, correct any that are genuinely mislabelled, and report both the
before and after numbers.

**Computational efficiency.** `train_compare.py` records tuning time and
inference time per model, and with no model significantly more accurate than
any other (see Comment 2), this becomes the deciding criterion rather than a
footnote. CART tunes in 2.2 s and predicts 1,999 pixels in 0.25 ms; LightGBM
takes 1,127 s to tune — 500× longer — for an accuracy difference of 0.9
percentage points that does not survive significance testing. SVM has the
highest accuracy but the slowest inference (20.6 ms), which matters if the
method is meant to scale to repeated city-wide mapping.

**Generalisability to other cities.** Take the Taipei-trained model and apply
it to a second city with similar spectral character — Kaohsiung or Taichung
would be defensible, or New Taipei City since it surrounds the study area. Even
a qualitative transfer result with a modest validation sample answers the
question. Alternatively, argue the limits explicitly from the training-sample
geography rather than claiming transferability you haven't tested.

**Urban planning and disaster management implications.** Tie the built-up
fraction to Taipei's flood exposure using the SRTM data already in Figure 1 —
impervious cover below a given elevation contour, intersected with the Keelung
and Tamsui floodplains. This is one extra GEE reduction and it turns a generic
closing paragraph into a result.

**Limitations.** Needs its own subsection covering: the 100-point validation
sample and its effect on per-class estimates; single-season imagery, which
conflates deciduous phenology with the plantation class and likely explains
part of the plantation/forest confusion; pixel-based classification without
texture or object-based context in a city with sub-10 m built structures; the
L1C product used without atmospheric correction to L2A; and the absence of an
independent reference dataset (the reference points were interpreted from the
same imagery being classified, which inflates agreement).

---

## Part B — Problems found in the actual GEE script

Reviewing the submitted RF script changed the priority order. These were not
raised by the editor, but they are more serious than what was raised.

### B0. The reported accuracies are inflated (train/test leakage)

The script does this:

```js
var classifier = ee.Classifier.smileRandomForest(10).train({
  features: classifierTraining, ...   // ALL samples
});
var trainingTesting = classifierTraining.randomColumn();
var testingSet = trainingTesting.filter(ee.Filter.greaterThanOrEquals('random', 0.8));
var confusionMatrix = ee.ConfusionMatrix(testingSet.classify(classifier)...);
```

`trainingSet` is created and then never used. The classifier is fitted on the
complete sample set, and the "test" 20% is a subset of the data it already saw.
Every accuracy figure in the paper — 90% / 83% / 71% and the three Kappa values
— is a resubstitution accuracy, not a validation accuracy. They will fall when
measured properly, probably by several points, and RF will fall furthest
because a 10-tree forest memorises training pixels readily.

This has to be fixed before anything else, because every number in the
Abstract, Results, Discussion and Conclusions derives from it.

### B1. Pixel-level splitting of polygon-derived samples

`sampleRegions` on training polygons returns every pixel inside each polygon.
Splitting those pixels at random puts near-identical neighbours on both sides
of the divide, which inflates accuracy even after B0 is fixed. The corrected
script assigns the random number at the polygon level so that all pixels from a
polygon fall on one side.

This also contradicts the Methods text, which claims "a minimum-distance rule
was applied during sample placement to avoid spatial autocorrelation". No such
rule exists in the code. That sentence must be removed or the rule actually
implemented.

**Subtlety found on download.** `ROI_TAIPEI_01` contains **five features** —
one MultiPolygon per class — together covering **110 separately drawn areas**:

| Class | Parts | Area (ha) | Approx. pixels @10 m |
|---|---|---|---|
| Water | 13 | 17.0 | 1,703 |
| Built-up | 25 | 20.0 | 1,999 |
| Forest | 45 | 33.5 | 3,347 |
| Plantation | 18 | 9.2 | 923 |
| Swamp | 9 | 4.9 | 491 |
| **Total** | **110** | **84.6** | **~8,460** |

Calling `randomColumn` on the collection as stored gives one random number per
*class*, not per polygon, so a threshold split sends entire classes to one side.
Both the JS and the Python path now explode the MultiPolygons into their 110
parts before splitting. Python additionally assigns the split per class, since
with a single global threshold there is a ~4% chance all nine swamp parts land
in train.

Two consequences for the manuscript. The sample-design table should report 110
training areas totalling ~85 ha and ~8,500 pixels, not 5 polygons. And swamp
gets only 2–3 validation polygons under a 70/30 split, which is the real reason
its producer's accuracy is unstable across tables — state that in Limitations,
and preferably draw more swamp polygons before the final run.

### B2. Methods text does not match the code

| Manuscript says | Script actually does |
|---|---|
| "summer 2020" imagery | `filterDate('2020-01-01','2020-12-31')` — full year median |
| "scenes with minimum cloud cover" | `CLOUDY_PIXEL_PERCENTAGE < 20` |
| B3, B4, B8 (Table 1, bold) | `['B2','B3','B4','B8']` — includes Blue |
| "100 randomly distributed reference points" | 20% of all polygon pixels — thousands of rows |
| Level-1C via `COPERNICUS/S2` | collection now deprecated |

The 100-point claim is the one to resolve first: the confusion matrices in the
paper sum to exactly 100, but this script cannot produce a 100-row test set.
Either the matrices came from a separate procedure that is not described, or
they were rescaled. Whichever it is, the Methods section must describe what was
actually done.

### B3. Other script issues

The export is named `Taiwan_1990_RF_New` — leftover from a different project,
and "1990" is wrong for 2020 data. `Export.image.toDrive` has no `maxPixels`,
which will fail at 10 m over the full city. The commented-out SVM block samples
at `scale: 30` while the RF block uses `scale: 10`; if the published SVM run
used that block, the two classifiers were trained on different sample sets,
which explains the mismatched confusion-matrix totals below.

---

## Part C — Internal inconsistencies in the manuscript

**The three confusion matrices do not share a reference set.** Column totals
for the "true" classes should be identical across Tables 3, 4 and 5 if the same
100 reference points were used. They aren't:

| Class | CART | SVM | RF |
|---|---|---|---|
| Water | 11 | 9 | 6 |
| Built-up | 22 | 34 | 33 |
| Forest | 35 | 46 | 41 |
| Plantation | 14 | 6 | 12 |
| Swamp | 18 | 5 | 8 |

Given B3 above, the likeliest cause is that the classifiers were run at
different sampling scales (30 m for SVM, 10 m for RF), producing different
sample sets. Either way the three accuracies are not directly comparable, and
McNemar's test cannot be applied to them — McNemar requires paired predictions
on identical samples. The corrected script builds one sample set, one split,
and exports paired predictions for all four classifiers, which resolves this by
construction.

**The area base is 292.8 sq.km, not 272 sq.km.** Back-calculating from every
area/percentage pair in the Results gives a total of ~292.8 sq.km, but the
Study Area section states 272 sq.km. A 7.6% discrepancy suggests the AOI
polygon includes a buffer or uses New Taipei boundaries at the margin. Check
the boundary asset and reconcile.

**Built-up area, RF.** Abstract says 100.97 sq.km; Results, Discussion and
Conclusions all say 100.94.

**Swamp share, RF.** Results give 18.47 sq.km (6.31%); the Discussion says
"swamp (3%)". The 3% figure is wrong.

**SVM water and swamp percentages are swapped.** Water is listed as 16.77
sq.km / 5.78% and swamp as 16.92 sq.km / 5.73%, but 16.77/292.8 = 5.73% and
16.92/292.8 = 5.78%.

**Unsupported claim in the Conclusions.** "Over 70% of the low-lying basin
terrain is already covered by impervious surfaces" — no elevation-stratified
analysis appears anywhere in the paper. Either compute it (see the flood
exposure suggestion above, which would support it) or remove the sentence.

**Feature set.** Only B3, B4 and B8 were used. Excluding B11/B12 is the most
likely single cause of the built-up↔swamp and built-up↔water confusion that
dominates the error analysis, since SWIR is what separates impervious surfaces
from water. Running the extended feature set as an ablation
(`gee_export_samples.py --extended`) would let you show this directly, and it
converts a methodological weakness into a contribution.

**Reference [breiman1988] has a mismatched DOI** — it points to a Japanese
journal (`10.20816/jalps...`), not the JASA paper. Check the rest of the
bibliography for similar copy-paste errors.

---

## Part C — Suggested order of work

Roughly two weeks of effort, front-loaded on the data problem:

1. Import repo and assets, get `taipei_samples.csv` on disk. (Day 1) **Done.**
2. Resolve the reference-set problem and enlarge the validation sample to
   ~500–750 stratified points. This blocks everything statistical, so do it
   first. (Days 2–4)

   Concretely: `train_compare.py` → `classify_map.py` → `draw_reference_sample.py`,
   then interpret the 715 points, then `olofsson.py --reference ...`. The
   interpretation is the long pole; start it before anything else and let the
   modelling run alongside.
3. Run tuning and the extended model set, both feature configurations. (Days 5–6)
4. McNemar and bootstrap CIs; area-weighted accuracies and area CIs from
   `olofsson.py`; build the revised tables. (Day 7)
5. Misclassification analysis and spectral separability figures. (Days 8–9)
6. Flood-exposure overlay for the planning implications. (Day 10)
7. Rewrite Methods, Results, Discussion; add the Limitations subsection. (Days 11–13)
8. Public repo, Zenodo DOI, new Data Availability Statement, Responses file. (Day 14)
