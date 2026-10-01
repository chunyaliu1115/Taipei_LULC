/**
 * Export the twelve administrative districts of Taipei City as GeoJSON.
 *
 * Needed for the fig3a-style map: the published Figure 3a draws each district
 * outline as a thin white line and labels it, and nothing in data/raw/ goes
 * below the city outline.
 *
 * Run this in the Earth Engine Code Editor, then Tasks -> Run. The file lands
 * in your Drive folder 'GEE_exports'; move it to
 *
 *     data/raw/taipei_districts.geojson
 *
 * Two sources are tried, in this order. Run the first cell, look at the print
 * output in the Console, and export whichever one returns 12 features with
 * usable names.
 */

var taipei = ee.Feature(
    ee.FeatureCollection('FAO/GAUL/2015/level1')
      .filter(ee.Filter.eq('ADM1_NAME', 'Taipei City'))
      .first());

// ---------------------------------------------------------------------------
// Source A - geoBoundaries ADM2, the one that actually carries Taiwan's
// district level. GAUL stops at ADM1 for Taiwan in most builds, so this is
// normally the one that works.
// ---------------------------------------------------------------------------
var adm2 = ee.FeatureCollection(
    'projects/sat-io/open-datasets/geoboundaries/CGAZ_ADM2');

var districtsA = adm2
    .filterBounds(taipei.geometry())
    // A district that merely touches the city edge is not a district of the
    // city. Requiring the centroid to fall inside drops New Taipei's ring of
    // surrounding districts, which otherwise come through on filterBounds.
    .map(function (f) {
      return f.set('inside',
          taipei.geometry().contains(f.geometry().centroid(100), 100));
    })
    .filter(ee.Filter.eq('inside', true));

print('Source A - geoBoundaries ADM2, feature count:', districtsA.size());
print('Source A - first feature (check the name property):',
      districtsA.first());

// ---------------------------------------------------------------------------
// Source B - FAO GAUL level 2, in case your GEE account resolves Taiwan's
// second level. Usually returns 0 features; check before using.
// ---------------------------------------------------------------------------
var districtsB = ee.FeatureCollection('FAO/GAUL/2015/level2')
    .filter(ee.Filter.eq('ADM1_NAME', 'Taipei City'));

print('Source B - GAUL level 2, feature count:', districtsB.size());

// ---------------------------------------------------------------------------
// Export. Change `chosen` to districtsB if that is the one with 12 features.
// The property list is deliberately wide: different builds of geoBoundaries
// name the label column shapeName, ADM2_NAME or TOWNENG, and asking for a
// property that does not exist is not an error in a table export - the column
// simply comes back empty, which is easier to debug than a failed task.
// ---------------------------------------------------------------------------
var chosen = districtsA;

Map.centerObject(taipei, 11);
Map.addLayer(taipei, {color: 'black'}, 'Taipei City');
Map.addLayer(chosen, {color: 'red'}, 'districts');

Export.table.toDrive({
  collection: chosen,
  description: 'taipei_districts',
  folder: 'GEE_exports',
  fileNamePrefix: 'taipei_districts',
  fileFormat: 'GeoJSON',
  selectors: ['shapeName', 'ADM2_NAME', 'TOWNENG', 'TOWNNAME', 'NAME_2']
});
