# Hello Reshape SSD Node.js Sample

Models with only 1 input and output are supported.

Make sure that static files are downloaded, you can prepare them by run:
```bash
node ../fetch-samples-assets.js
```

Run sample:
```bash
node hello_reshape_ssd.js ../../assets/models/road-segmentation-adas-0001.xml ../../assets/images/empty_road_mapillary.jpg AUTO
```
Where
```bash
node hello_reshape_ssd.js *path_to_model_file* *path_to_img* *device*
```

Other details see in [../../../python/hello_reshape_ssd/README.md](../../../python/hello_reshape_ssd/README.md)

