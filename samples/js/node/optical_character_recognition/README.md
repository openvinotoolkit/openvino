# Optical Character Recognition Node.js Sample

Make sure that static files are downloaded, you can prepare them by run:
```bash
node ../fetch-samples-assets.js
```

Run sample:
```bash
node optical-character-recognition.js ../../assets/models/horizontal-text-detection-0001.xml ../../assets/models/text-recognition-resnet-fc.xml ../../assets/images/intel_rnb.jpg AUTO
```
Where:
```bash
node optical-character-recognition.js *path_to_detection_model_file* *path_to_recognition_model_file* *path_to_img* *device*
```
