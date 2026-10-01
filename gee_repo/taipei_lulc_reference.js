/**
 * Taipei City LULC classification - corrected reference script
 * Liu, ICCK Journal of Image Analysis and Processing (JIAP-2026-490298)
 *
 * Rewritten from the original submission script. Four substantive changes,
 * each flagged inline with [FIX]:
 *
 *   [FIX 1] The original trained the classifier on ALL samples and then
 *           evaluated it on a 20% subset of those same samples. The reported
 *           accuracies were resubstitution accuracies on seen data. Here the
 *           classifier is trained on the training split only.
 *   [FIX 2] The original split pixels at random. Pixels drawn from the same
 *           training polygon are near-duplicates, so a random pixel split puts
 *           near-identical rows on both sides. The split is now applied at the
 *           POLYGON level, so every pixel of a polygon lands in one side only.
 *           Note that ROI_TAIPEI_01 holds one MultiPolygon per class, so the
 *           polygons must be exploded into their 110 parts first - otherwise
 *           "polygon level" means "class level" and whole classes disappear
 *           from one side.
 *   [FIX 3] COPERNICUS/S2 is deprecated; switched to COPERNICUS/S2_HARMONIZED.
 *   [FIX 4] All four classifiers now share one split, and per-sample
 *           predictions are exported so McNemar's test can be run on paired
 *           predictions.
 *
 * Asset IDs below match the Imports panel of Taipei_City_2020_LULC_S2_RF.
 */

// ---------------------------------------------------------------------------
// 1. Inputs
// ---------------------------------------------------------------------------
// Legacy users/ paths, as they appear in the Code Editor Imports panel.
// Note the _01 suffix on the ROI asset.
var select_feature = ee.FeatureCollection('users/amanjmi01/Taipei_City');
var polygons       = ee.FeatureCollection('users/amanjmi01/ROI_TAIPEI_01');

var CLASS_PROP = 'Id';          // classes 1..5 in your assets
var SCALE      = 10;
var SEED       = 42;
var TRAIN_FRAC = 0.7;

// Kept identical to the submitted script so results stay comparable.
var DATE_START = '2020-01-01';
var DATE_END   = '2020-12-31';
var MAX_CLOUD  = 20;
var bands = ['B2', 'B3', 'B4', 'B8'];

// ---------------------------------------------------------------------------
// 2. Composite
// ---------------------------------------------------------------------------
function maskS2clouds(image) {
  var qa = image.select('QA60');
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
               .and(qa.bitwiseAnd(cirrusBitMask).eq(0));
  return image.updateMask(mask).divide(10000)
              .copyProperties(image, ['system:time_start']);
}

// [FIX 3] COPERNICUS/S2 -> COPERNICUS/S2_HARMONIZED
var dataset = ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
  .filterDate(DATE_START, DATE_END)
  .filterBounds(select_feature)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', MAX_CLOUD))
  .map(maskS2clouds);

var s2_Taipei = dataset.median().clip(select_feature);

// Extended feature set for the ablation the revision needs. SWIR is what
// separates impervious surfaces from water, and its absence is the most likely
// cause of the built-up/swamp confusion in the submitted results.
var ndvi  = s2_Taipei.normalizedDifference(['B8', 'B4']).rename('NDVI');
var ndbi  = s2_Taipei.normalizedDifference(['B11', 'B8']).rename('NDBI');
var mndwi = s2_Taipei.normalizedDifference(['B3', 'B11']).rename('MNDWI');
var bandsExt = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12'];
var inputExt = s2_Taipei.select(bandsExt).addBands([ndvi, ndbi, mndwi]);

var input = s2_Taipei.select(bands);     // swap to inputExt for the ablation

Map.centerObject(select_feature);
Map.addLayer(s2_Taipei, {min: 0, max: 0.3, bands: ['B8', 'B4', 'B3']},
             'Sentinel-2 False Colour', false);

// ---------------------------------------------------------------------------
// 3. Polygon-level split  [FIX 2]
// ---------------------------------------------------------------------------
// ROI_TAIPEI_01 contains five features - one MultiPolygon per class, covering
// 110 separately drawn areas. randomColumn() on that gives five random numbers,
// one per class, so a threshold split would send entire classes to one side.
// Explode into single polygons first.
var explode = function (feat) {
  feat = ee.Feature(feat);
  var geom = feat.geometry();
  var parts = ee.List(ee.Algorithms.If(
    ee.String(geom.type()).compareTo('MultiPolygon').eq(0),
    geom.coordinates().map(function (c) {
      return ee.Geometry.Polygon(ee.List(c), null, false);
    }),
    ee.List([geom])
  ));
  var props = feat.toDictionary([CLASS_PROP, 'Class']);
  return ee.FeatureCollection(parts.map(function (g) {
    return ee.Feature(ee.Geometry(g), props);
  }));
};

var parts = ee.FeatureCollection(polygons.map(explode)).flatten();
print('ROI features:', polygons.size(), '-> single polygons:', parts.size());

// Give every PART a stable random number, then carry it onto each sampled
// pixel. Splitting on this keeps all pixels of a part on one side.
//
// CAVEAT: a single global threshold is only probabilistically fair to the
// minority classes - with nine swamp parts there is a ~4% chance all nine land
// in train. The Python pipeline (src/gee_export_samples.py) assigns the split
// per class instead, which removes the chance entirely. Prefer that route for
// the numbers that go in the paper; this script is the in-cloud reference.
var partsRnd = parts.randomColumn('poly_rnd', SEED);

var samples = input.sampleRegions({
  collection: partsRnd,
  properties: [CLASS_PROP, 'poly_rnd'],
  scale: SCALE,
  tileScale: 4,
  geometries: true
}).map(function (f) {
  var isTrain = ee.Number(f.get('poly_rnd')).lt(TRAIN_FRAC);
  return f.set('split', ee.Algorithms.If(isTrain, 'train', 'val'));
});

var trainingSet = samples.filter(ee.Filter.eq('split', 'train'));
var testingSet  = samples.filter(ee.Filter.eq('split', 'val'));

print('Total samples:', samples.size());
print('Training samples:', trainingSet.size());
print('Testing samples:', testingSet.size());
print('Per-class training counts:', trainingSet.aggregate_histogram(CLASS_PROP));
print('Per-class testing counts:',  testingSet.aggregate_histogram(CLASS_PROP));
print('Parts per class:', parts.aggregate_histogram(CLASS_PROP));
// Check this last one: any class with 0 in the testing histogram means the
// threshold split failed for it. Re-run with a different SEED or use Python.

// ---------------------------------------------------------------------------
// 4. Classifiers - all trained on the SAME training split  [FIX 1, FIX 4]
// ---------------------------------------------------------------------------
var classifiers = {
  CART: ee.Classifier.smileCart(),
  SVM: ee.Classifier.libsvm({kernelType: 'RBF', gamma: 0.5, cost: 10}),
  RF: ee.Classifier.smileRandomForest({numberOfTrees: 500, seed: SEED}),
  // Closest native analogue to XGBoost/LightGBM available inside GEE.
  // Call it "gradient tree boosting", NOT XGBoost - it is a different library.
  GTB: ee.Classifier.smileGradientTreeBoost({
    numberOfTrees: 300, shrinkage: 0.05, samplingRate: 0.8, seed: SEED
  })
};

var lulcPalette = [
  '1200ff', // 1 Water        [BLUE]
  'ff0000', // 2 Built-up     [RED]
  '2a7f00', // 3 Forest/Veg   [DARK GREEN]
  '96b058', // 4 Plantation   [GREEN]
  '04d1ff'  // 5 Swamp        [SKY]
];

var predictionTable = testingSet;   // accumulates one column per classifier

Object.keys(classifiers).forEach(function (name) {

  // [FIX 1] train on trainingSet, NOT on the full sample set
  var trained = classifiers[name].train({
    features: trainingSet,
    classProperty: CLASS_PROP,
    inputProperties: bands
  });

  var classified = input.classify(trained);
  Map.addLayer(classified, {palette: lulcPalette, min: 1, max: 5},
               name + ' classification', name === 'RF');

  var scored = testingSet.classify(trained);
  var cm = scored.errorMatrix(CLASS_PROP, 'classification');

  print('=== ' + name + ' ===');
  print(name + ' confusion matrix:', cm);
  print(name + ' overall accuracy:', cm.accuracy());
  print(name + ' kappa:', cm.kappa());
  print(name + " producer's accuracy:", cm.producersAccuracy());
  print(name + " consumer's accuracy:", cm.consumersAccuracy());

  // Class areas in sq.km - lets you check the 292.8 vs 272 discrepancy
  var areas = ee.Image.pixelArea().divide(1e6).addBands(classified).reduceRegion({
    reducer: ee.Reducer.sum().group({groupField: 1, groupName: 'class'}),
    geometry: select_feature.geometry(),
    scale: SCALE,
    maxPixels: 1e13
  });
  print(name + ' class areas (sq.km):', areas);

  // [FIX 4] keep each classifier's prediction on the SAME test rows
  var preds = scored.select(['classification'], ['pred_' + name]);
  predictionTable = ee.Join.inner('a', 'b').apply(
    predictionTable, preds,
    ee.Filter.equals({leftField: 'system:index', rightField: 'system:index'})
  ).map(function (f) {
    return ee.Feature(f.get('a')).copyProperties(ee.Feature(f.get('b')));
  });

  Export.image.toDrive({
    image: classified.toByte(),
    description: 'taipei_lulc_' + name,
    folder: 'GEE_exports',
    region: select_feature.geometry(),
    scale: SCALE,
    maxPixels: 1e13
  });
});

// ---------------------------------------------------------------------------
// 5. Exports for the local analysis
// ---------------------------------------------------------------------------

// (a) Full sample table -> feeds XGBoost / LightGBM / MLP and the tuning search
Export.table.toDrive({
  collection: samples,
  description: 'taipei_samples',
  folder: 'GEE_exports',
  fileFormat: 'CSV'
});

// (b) Paired predictions on the shared test set -> feeds McNemar's test
Export.table.toDrive({
  collection: predictionTable,
  description: 'taipei_gee_predictions',
  folder: 'GEE_exports',
  fileFormat: 'CSV'
});
