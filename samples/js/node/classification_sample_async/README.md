# Image Classification Async Node.js Sample

Models with only 1 input and output are supported.

Run:
```bash
node classification_sample_async.js -m ../../assets/models/v3-small_224_1.0_float.xml -i ../../assets/images/coco.jpg -i ../../assets/images/coco_hollywood.jpg -d AUTO
```

Where
```bash
node classification_sample_async.js -m *path_to_model_file* -i *path_to_img1* -i *path_to_img2* -d *device*
```

Other details see in [../../../python/classification_sample_async/README.md](../../../python/classification_sample_async/README.md)
