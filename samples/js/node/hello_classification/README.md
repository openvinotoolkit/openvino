# Hello Classification Node.js Sample

Models with only 1 input and output are supported.

Make sure that static files are downloaded, you can prepare them by run:
```bash
node ../fetch-samples-assets.js
```

Run sample:
```bash
node hello_classification.js ../../assets/models/v3-small_224_1.0_float.xml ../../assets/images/coco_hollywood.jpg AUTO
```
Where
```bash
node hello_classification.js *path_to_model_file* *path_to_img* *device*
```

Other details see in [../../../python/hello_classification/README.md](../../../python/hello_classification/README.md)
