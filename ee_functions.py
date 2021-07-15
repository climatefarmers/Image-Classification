# -*- coding: utf-8 -*-


'''
List of functions used in the corresponding notebook EE_functions_and_viz.ipynb
'''

import ee
import geopandas as gpd
import pprint
import folium
import numpy as np
import pandas as pd
import geopandas as gpd

def read_shape_file(filePath: str) -> ee.Geometry.Polygon:


  '''
  Reads the passed shape file anad extracts the boundaries of the contained farm(s) as ee.Polygon(s).

  regionsOfInterest = read_shape_file(path_to_shapefile)

  '''

  # make sure that geopandas and numpy are imported
  try:
    import geopandas as gpd
    import numpy as np
    import warnings

  except ImportError:
    print('This function relies on external libraries. Consult the documentation and install'
      'the necessary libraries.')
  
  gdf = gpd.read_file(filePath)

  # converting the projection to the standard WGS:84 projection
  if gdf.crs:
    gdf = gdf.to_crs('EPSG:4326')
  else:
    gdf = gdf.set_crs('EPSG:4326')

  farmBoundaries = gdf['geometry']

  allCoordinates = []
  for b in farmBoundaries.boundary:
    if b.geom_type == 'Point' or b.geom_type == 'MultiPoint':
      warnings.warn(warnMessage)
      
    elif b.geom_type == 'MultiLineString':
      for geom in range(len(b.geoms)):

        # extract the x and y coordinates for each region in the file and form a list of tuples [(x-coord, y-coord)]
        x,y = b[geom].coords.xy
        xy = list(zip(x,y))
        allCoordinates.append(xy)
    else:
      x,y = b.coords.xy
      xy = list(zip(x,y))
      allCoordinates.append(xy)
    
    for coordinate in allCoordinates:
      coordinate = ee.Geometry.Polygon(coordinate)
  return ee.Geometry.Polygon(allCoordinates)

def get_sentinel_and_cloudless_collections(aoi: ee.Geometry.Polygon, startDate: str, endDate: str) -> ee.ImageCollection:

  '''
  Joins the S2_SR and s2cloudless image collections. This function makes it possible
  to look at the probability of a pixel in the image being a cloud.

  imageCollection = get_sentinel_and_cloudless_collections(roi, '2020-01-01', '2021-01-01')
  '''

  # Import and filter Sentinel-2 images
  sentinelCollection = (ee.ImageCollection('COPERNICUS/S2_SR')
      .filterBounds(aoi)
      .filterDate(startDate, endDate)
      .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', CLOUD_FILTER)))

  # Import and filter the s2cloudless (cloud probability) dataset
  s2cloudlessCollection = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
      .filterBounds(aoi)
      .filterDate(startDate, endDate))

  # Join the filtered s2cloudless collection to the sentinel-2 collection by the 'system:index' property
  return ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
      'primary': sentinelCollection,
      'secondary': s2cloudlessCollection,
      'condition': ee.Filter.equals(**{
          'leftField': 'system:index',
          'rightField': 'system:index'
      })
  }))

def add_cloud_bands(img: ee.image.Image) -> ee.image.Image:

    '''
    Adds a cloud probability layer to the passed image.

    imageWithCloudProbabilityLayer = add_cloud_bands(someImage)
    '''

    # Get s2cloudless image, subset the probability band.
    cloudProbability = ee.Image(img.get('s2cloudless')).select('probability')

    # Condition s2cloudless by the probability threshold value.
    isCloud = cloudProbability.gt(CLD_PRB_THRESH).rename('clouds')

    # Add the cloud probability layer and cloud mask as image bands.
    return img.addBands(ee.Image([cloudProbability, isCloud]))

def add_shadow_bands(img: ee.image.Image) -> ee.image.Image:

    '''
    Adds a cloud shadow probability layer to the passed image.

    imageWithCloudShadowProbabilityLayer = add_shadow_bands(someImage)
    '''

    # Identify water pixels from the SCL band.
    notWater = img.select('SCL').neq(6)

    # Identify dark NIR pixels that are not water (potential cloud shadow pixels).
    SR_BAND_SCALE = 1e4
    darkPixels = img.select('B8').lt(NIR_DRK_THRESH*SR_BAND_SCALE).multiply(notWater).rename('dark_pixels')

    # Determine the direction to project cloud shadow from clouds (assumes UTM projection).
    shadowAzimuth = ee.Number(90).subtract(ee.Number(img.get('MEAN_SOLAR_AZIMUTH_ANGLE')));

    # Project shadows from clouds for the distance specified by the CLD_PRJ_DIST input.
    cloudProjection = (img.select('clouds').directionalDistanceTransform(shadowAzimuth, CLD_PRJ_DIST*10)
        .reproject(**{'crs': img.select(0).projection(), 'scale': 100})
        .select('distance')
        .mask()
        .rename('cloud_transform'))

    # Identify the intersection of dark pixels with cloud shadow projection.
    shadows = cloudProjection.multiply(darkPixels).rename('shadows')

    # Add dark pixels, cloud projection, and identified shadows as image bands.
    return img.addBands(ee.Image([darkPixels, cloudProjection, shadows]))


def add_cloud_shadow_mask(img: ee.image.Image) -> ee.image.Image:

    '''
    Adds a mask to remove clouds and cloud shadows from the passed image.

    maskedImage = add_cloud_shadow_mask(someImage)
    '''

    # Add cloud component bands.
    imgCloud = add_cloud_bands(img)

    # Add cloud shadow component bands.
    imgCloudShadow = add_shadow_bands(imgCloud)

    # Combine cloud and shadow mask, set cloud and shadow as value 1, else 0.
    isCloudShadow = imgCloudShadow.select('clouds').add(imgCloudShadow.select('shadows')).gt(0)

    # Remove small cloud-shadow patches and dilate remaining pixels by BUFFER input.
    # 20 m scale is for speed, and assumes clouds don't require 10 m precision.
    isCloudShadow = (isCloudShadow.focal_min(2).focal_max(BUFFER*2/20)
        .reproject(**{'crs': img.select([0]).projection(), 'scale': 20})
        .rename('cloudmask'))

    # Add the final cloud-shadow mask to the image.
    return imgCloudShadow.addBands(isCloudShadow)


def apply_cloud_shadow_mask(img: ee.image.Image) -> ee.image.Image:


    # Subset the cloudmask band and invert it so clouds/shadow are 0, else 1.
    notCloudShadow = img.select('cloudmask').Not()

    # Subset reflectance bands and update their masks, return the result.
    return img.select('B.*','NDVI','NDTI','probability','clouds','cloudmask','cloud_transform','brightnessIndex','dark_pixels','shadows','SoilColor').updateMask(notCloudShadow)


# Define a method for displaying Earth Engine image tiles to a folium map.
def add_ee_layer(self, ee_image_object, vis_params, name, show=True, opacity=1, min_zoom=0):
    map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
    folium.raster_layers.TileLayer(
        tiles=map_id_dict['tile_fetcher'].url_format,
        attr='Map Data &copy; <a href="https://earthengine.google.com/">Google Earth Engine</a>',
        name=name,
        show=show,
        opacity=opacity,
        min_zoom=min_zoom,
        overlay=True,
        control=True
        ).add_to(self)

# Add the Earth Engine layer method to folium.
folium.Map.add_ee_layer = add_ee_layer

def add_NDVI_band(image: ee.image.Image) -> ee.image.Image:

  '''
  Calculates the NDVI value for each pixel in the passed image and adds an NDVI band to the passed image.

  imageWithNDVIband = add_NDVI_band(someImage)
  '''

  ndvi = image.normalizedDifference(['B5', 'B4']).rename('NDVI');
  return image.addBands(ndvi)

def add_NDTI_band(image: ee.image.Image) -> ee.image.Image:

  '''
  Calculates the NDTI value for each pixel in the passed image and adds an NDTI band to the passed image.

  imageWithNDTIband = add_NDTI_band(someImage)
  '''

  ndti = image.normalizedDifference(['B11', 'B12']).rename('NDTI')
  return image.addBands(ndti)


def add_soil_color_band(image: ee.image.Image) -> ee.image.Image:

  '''
  Calculates the soil color value for each pixel in the passed image and adds a soil color band to the passed image.

  imageWithSoilColorBand = add_soil_color_band(someImage)
  '''

  soilColor = image.normalizedDifference(['B4', 'B3']).rename('SoilColor')
  return image.addBands(soilColor)

def add_soil_brightness_band(image: ee.image.Image) -> ee.image.Image:

  '''
  Calculates the brightness value for each pixel in the passed image and adds a brightness band to the passed image.

  imageWithSoilBrightnessBand = add_soil_brightness_band(someImage)
  '''

  bi = image.expression(
    '(((GREEN**2 + RED**2 + NIR**2)**(0.5)) / 10**4)', {
      'GREEN': image.select('B3'),
      'RED': image.select('B4'),
      'NIR': image.select('B8')
}).rename('brightnessIndex')

  return image.addBands(bi)


def get_collection_means(image: ee.image.Image, index: str, geometry: ee.geometry.Geometry) -> ee.ImageCollection:
 
  '''
  Computes the mean of the passed index over the passed image.
  The value is a dictionary, so get the index value from the dictionary.


  '''
  value = image.reduceRegion(**{
    'geometry': geometry,
    'reducer': ee.Reducer.mean(),
  }).get(index)
 
  # Adding computed index value
  newFeature = ee.Feature(None, {
      index : value
  }).copyProperties(image, [
      'system:time_start',
      'SUN_ELEVATION'
  ])
  return newFeature

def display_cloud_layers(col: ee.ImageCollection, aoi: ee.Geometry.Polygon, clipped: int):

  '''
  Creates a folium map and visualizes the various cloud layer bands. Can display the entire image
  or only the region of interest using the clipped parameter.

  display_cloud_layers(someCollection, roi, 1)
  '''
    
  if clipped == 1:
    img = col.first().clip(aoi)
  else:
    img = col.first()

  # Subset layers and prepare them for display.
  clouds = img.select('clouds').selfMask()
  shadows = img.select('shadows').selfMask()
  dark_pixels = img.select('dark_pixels').selfMask()
  probability = img.select('probability')
  cloudmask = img.select('cloudmask').selfMask()
  cloud_transform = img.select('cloud_transform')

  # Create a folium map object.
  center = aoi.centroid(10).coordinates().reverse().getInfo()
  m = folium.Map(location=center, zoom_start=12)

  # Add layers to the folium map.
  m.add_ee_layer(img,
                  {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 2500, 'gamma': 1.1},
                  'S2 image', True, 1, 9)
  m.add_ee_layer(probability,
                  {'min': 0, 'max': 100},
                  'probability (cloud)', False, 1, 9)
  m.add_ee_layer(clouds,
                  {'palette': 'e056fd'},
                  'clouds', False, 1, 9)
  m.add_ee_layer(cloud_transform,
                  {'min': 0, 'max': 1, 'palette': ['white', 'black']},
                  'cloud_transform', False, 1, 9)
  m.add_ee_layer(dark_pixels,
                  {'palette': 'orange'},
                  'dark_pixels', False, 1, 9)
  m.add_ee_layer(shadows, {'palette': 'yellow'},
                  'shadows', False, 1, 9)
  m.add_ee_layer(cloudmask, {'palette': 'orange'},
                  'cloudmask', True, 0.5, 9)

  # Add a layer control panel to the map.
  m.add_child(folium.LayerControl())

  # Display the map.
  display(m)

def reduce_region_to_mean(image: ee.image.Image ,roi: ee.geometry.Geometry) -> dict:

  '''
  Returns the mean pixel value of the passed region of interest.

  meanOfPassedRegion = reduce_region_to_mean(someImage, roi)
  '''

  meansDict = image.reduceRegion(**{    
      'reducer' : ee.Reducer.mean(),
      'geometry': roi,   # our defined region of interest
      'scale': 10    # scale = 10 because the resolution of Sentinel-2 bands of interest are at 10m/pixel
  })
  return meansDict


def get_mean(image: ee.image.Image, roi: ee.geometry.Geometry, index: str) -> float:

  '''
  Returns the mean pixel value of the desired image band of the region of interest.

  meanIndexValueOfPassedRegion = get_mean(someImage, roi, 'NDTI')
  '''

  return reduce_region_to_mean(image,roi).get(index)#.getInfo()
  

def reduce_region_to_median(image: ee.image.Image ,roi: ee.geometry.Geometry) -> dict:

  '''
  Returns the median pixel value of the passed region of interest.

  medianOfPassedRegion = reduce_region_to_median(someImage, roi)
  '''

  mediansDict = image.reduceRegion(**{    
      'reducer' : ee.Reducer.median(),
      'geometry': roi,   # our defined region of interest
      'scale': 10    # scale = 10 because the resolution of Sentinel-2 bands of interest are at 10m/pixel
  })
  return mediansDict


def get_median(image: ee.image.Image, roi: ee.geometry.Geometry, index: str) -> float:

  '''
  Returns the median pixel value of the desired image band of the region of interest.

  medianIndexValueOfPassedRegion = get_median(someImage, roi, 'NDTI')
  '''

  return reduce_region_to_median(image,roi).get(index)#.getInfo()


def get_date(image: ee.image.Image):

  '''
  Fetches the date of the passed image.

  date = get_date(someImage)
  '''

  unixDate = image.get('system:time_start')#.getInfo()
  return ee.Date(unixDate).format('YYYY-MM-dd')#.getInfo()


def isValidImage(image: ee.image.Image, roi: ee.geometry.Geometry, cloudThreshold: int, pixelThreshold: int) -> bool:

  '''
  Classifies whether an image should be included for analysis based on the number pixels in a region of interest
  that are likely to be clouds.

  isValidImage(someImage, roi, 30, 30)
  '''

  pixels = getValidPixels(image,roi,cloudThreshold)
  if (len(pixels[1])/len(pixels[0]) * 100) >= pixelThreshold:
    return True
  else:
    return False


def get_valid_pixels(image: ee.image.Image, roi: ee.geometry.Geometry, cloudThreshold: int) -> tuple:

  '''
  Returns a 1-D array of pixels from the image and the pixels that have a low probability of being a cloud.

  allPixels, validPixels = get_valid_pixels(someImage, roi, 30)
  '''

  pixelsDict = (image
    .select('probability')
    .reduceRegion(**{
    'reducer': ee.Reducer.toList(), 
    'geometry': roi,
    'scale': 10
  }))

  # List of pixels, containing a list of band values
  pixels = ee.Array(pixelsDict.values()).transpose().toList() 

  validPixels = []
  for pixel in pixels.flatten().getInfo():
    if pixel < cloudThreshold:
      validPixels.append(pixel)

  return [pixels.flatten().getInfo(), validPixels]


def get_cloud_coverage_over_roi(image: ee.image.Image, roi: ee.geometry.Geometry) -> ee.image.Image:

  '''
  Adds a band to the passed image indicating the percentage of cloudy pixels in a region of interest.

  imageWithCloudyROIband = get_cloud_coverage_over_roi(someImage, roi)
  '''

  clouds = image.select('cloudmask').reduceRegion(**{
      'reducer': ee.Reducer.count(),
      'geometry': roi,
      'scale': 10
  }).get('cloudmask')

  npix = image.select('cloudmask').unmask().reduceRegion(**{
      'reducer': ee.Reducer.count(),
      'geometry': roi,
      'scale': 10
  }).get('cloudmask')

  cloudCoverageOverRegion = ee.Number(1).subtract(ee.Number(clouds).divide(npix)).multiply(100)

  return image.set('cloud_coverage_roi', cloudCoverageOverRegion)


def add_feature_properties_as_bands(image: ee.image.Image, roi: ee.geometry.Geometry) -> ee.image.Image:

  '''
  Adds below statistical properties of indices to the passed image as a separate image band.

  imageWithNewBands = add_feature_properties_as_bands(someImage, roi)
  '''

  image = image.set('AVERAGE_NDVI', get_mean(image,roi,'NDVI'))
  image = image.set('MEDIAN_NDVI', get_median(image,roi,'NDVI'))
  image = image.set('MEDIAN_NDTI', get_median(image,roi,'NDTI'))
  image = image.set('AVERAGE_NDTI', get_mean(image,roi,'NDTI'))
  image = image.set('AVERAGE_BRIGHTNESS', get_mean(image,roi,'brightnessIndex'))
  image = image.set('AVERAGE_SOILCOLOR', get_mean(image,roi,'SoilColor'))


  return image


def extract_time_series_features(collection: ee.ImageCollection) -> pd.DataFrame:

  '''
  Collects the listed image properties and stores them in a pandas DataFrame
  for later analysis.

  features = extract_time_series_feautres(someImageCollection)
  '''

  try:
    import pandas as pd
  except ImportError:
    print('This function relies on external libraries. Consult the documentation and install'
      'the necessary libraries.')
  
  #ee.Date(image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
  dates = collection.aggregate_array('system:time_start').getInfo()
  for i in range(len(dates)):
    dates[i] = ee.Date(dates[i]).format('YYYY-MM-dd').getInfo()
  avgNDVI = collection.aggregate_array('AVERAGE_NDVI').getInfo()
  medianNDVI = collection.aggregate_array('MEDIAN_NDVI').getInfo()
  avgNDTI = collection.aggregate_array('AVERAGE_NDTI').getInfo()
  medianNDTI = collection.aggregate_array('MEDIAN_NDTI').getInfo()
  avgBrightness = collection.aggregate_array('AVERAGE_BRIGHTNESS').getInfo()
  avgSoilColor = collection.aggregate_array('AVERAGE_SOILCOLOR').getInfo()
  cloudiness = collection.aggregate_array('cloud_coverage_roi').getInfo()

  featureDict = {
      'Date': dates,
      'Average_NDVI': avgNDVI,
      'Median_NDVI': medianNDVI,
      'Average_NDTI': avgNDTI,
      'Median_NDTI': medianNDTI,
      'Average_Brightness': avgBrightness,
      'Average_SoilColor': avgSoilColor,
      'Cloudiness': cloudiness
      }

  feature_df = pd.DataFrame([featureDict])
  feature_df = feature_df.apply(pd.Series.explode)

  return feature_df