# Benchmark Information

The benchmarks in this folder were tested using single input models with FP32 precision on the following models:
- [mobilenet-v3-small-1.0-224-tf](https://docs.openvino.ai/2023.3/omz_models_model_mobilenet_v3_small_1_0_224_tf.html)
- [text-recognition-resnet-fc](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/text-recognition-resnet-fc/README.md)
- [Multiclass Selfie-segmentation model](https://ai.google.dev/edge/mediapipe/solutions/vision/image_segmenter#multiclass-model)

# How to run
```sh
node <sample_name> <path_to_model>
```
e.g.
```sh
node asynchronous_benchmark.js ../selfie_multiclass_256x256.xml
```
