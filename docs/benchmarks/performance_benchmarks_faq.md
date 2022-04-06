# Performance Information Frequently Asked Questions {#openvino_docs_performance_benchmarks_faq}

The following questions and answers are related to [performance benchmarks](./performance_benchmarks.md) published on the documentation site.

#### 1. How often do performance benchmarks get updated?
New performance benchmarks are typically published on every `major.minor` release of the Intel® Distribution of OpenVINO™ toolkit.

#### 2. Where can I find the models used in the performance benchmarks?
All of the models used are included in the toolkit's [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo) GitHub repository.

#### 3. Will there be new models added to the list used for benchmarking?
The models used in the performance benchmarks were chosen based on general adoption and usage in deployment scenarios. We're continuing to add new models that support a diverse set of workloads and usage.

#### 4. What does CF or TF in the graphs stand for?
CF means Caffe*, while TF means TensorFlow*.

#### 5. How can I run the benchmark results on my own?
All of the performance benchmarks were generated using the open-sourced tool within the Intel® Distribution of OpenVINO™ toolkit called `benchmark_app`, which is available in both [C++](../../samples/cpp/benchmark_app/README.md) and [Python](../../tools/benchmark_tool/README.md).

#### 6. What image sizes are used for the classification network models?
The image size used in the inference depends on the network being benchmarked. The following table shows the list of input sizes for each network model.
|   **Model**                                                                                                                        |   **Public Network**               |     **Task**                | **Input Size** (Height x Width)   |
|------------------------------------------------------------------------------------------------------------------------------------|------------------------------------|-----------------------------|-----------------------------------|
| [bert-base-cased](https://github.com/PaddlePaddle/PaddleNLP/tree/v2.1.1)                                                           | BERT                               | question / answer           | 124                               |
| [bert-large-uncased-whole-word-masking-squad](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/bert-large-uncased-whole-word-masking-squad-int8-0001) | BERT-large  | question / answer | 384                   |
| [bert-small-uncased-whole-masking-squad-0002](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/bert-small-uncased-whole-word-masking-squad-0002) | BERT-small | question / answer        | 384  |
| [brain-tumor-segmentation-0001-MXNET](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/brain-tumor-segmentation-0001) | brain-tumor-segmentation-0001 | semantic segmentation       | 128x128x128 |
| [brain-tumor-segmentation-0002-CF2](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/brain-tumor-segmentation-0002)   | brain-tumor-segmentation-0002 | semantic segmentation       | 128x128x128 |
| [deeplabv3-TF](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/deeplabv3)                                    |  DeepLab v3 Tf                        | semantic segmentation      | 513x513                          |
| [densenet-121-TF](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/densenet-121-tf)                  | Densenet-121 Tf                        | classification              | 224x224                 |
| [efficientdet-d0](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/efficientdet-d0-tf)               | Efficientdet                          | classification | 512x512 |
| [facenet-20180408-102900-TF](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/facenet-20180408-102900)        | FaceNet TF                            | face recognition            | 160x160                        |
| [Facedetection0200](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/face-detection-0200)                | FaceDetection0200 | detection | 256x256 |
| [faster_rcnn_resnet50_coco-TF](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/faster_rcnn_resnet50_coco)    | Faster RCNN Tf                        | object detection            | 600x1024               |
| [forward-tacotron-duration-prediction](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/forward-tacotron) | ForwardTacotron | text to speech | 241 |
| [inception-v4-TF](https://github.com/openvinotoolkit/open_model_zoo/tree/develop/models/public/googlenet-v4-tf)          | Inception v4 Tf (aka GoogleNet-V4)    | classification              | 299x299          |
| [inception-v3-TF](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/googlenet-v3)                | Inception v3 Tf                       | classification              | 299x299          |
| [mask_rcnn_resnet50_atrous_coco](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/mask_rcnn_resnet50_atrous_coco) | Mask R-CNN ResNet50 Atrous | instance segmentation | 800x1365 |
| [mobilenet-ssd-CF](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/mobilenet-ssd)                  | SSD (MobileNet)_COCO-2017_Caffe       | object detection            | 300x300             |
| [mobilenet-v2-1.0-224-TF](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/mobilenet-v2-1.0-224)        | MobileNet v2 Tf                       | classification              | 224x224             |
| [mobilenet-v2-pytorch](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/mobilenet-v2-pytorch )      | Mobilenet V2 PyTorch                  | classification              | 224x224               |
| [Mobilenet-V3-small](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/mobilenet-v3-small-1.0-224-tf) | Mobilenet-V3-1.0-224 | classifier | 224x224 |
| [Mobilenet-V3-large](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/mobilenet-v3-large-1.0-224-tf) | Mobilenet-V3-1.0-224 | classifier | 224x224 |
| [pp-ocr-rec](https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.1/)                                                       | PP-OCR | optical character recognition | 32x640 |
| [pp-yolo](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.1)                                                     | PP-YOLO                                | detection | 640x640 |
| [resnet-18-pytorch](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/resnet-18-pytorch)                  | ResNet-18 PyTorch                     | classification              | 224x224             |
| [resnet-50-pytorch](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/resnet-50-pytorch)              | ResNet-50 v1 PyTorch                  | classification              | 224x224                        |
| [resnet-50-TF](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/resnet-50-tf)                  | ResNet-50_v1_ILSVRC-2012              | classification              | 224x224             |
| [yolo_v4-TF](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/yolo-v4-tf)                            | Yolo-V4 TF                            |  object detection          | 608x608                        |
| [ssd_mobilenet_v1_coco-TF](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/ssd_mobilenet_v1_coco)   | ssd_mobilenet_v1_coco                 | object detection            | 300x300                        |
| [ssdlite_mobilenet_v2-TF](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/ssdlite_mobilenet_v2)     | ssdlite_mobilenet_v2                  | object detection            | 300x300                        |
| [unet-camvid-onnx-0001](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/intel/unet-camvid-onnx-0001/description/unet-camvid-onnx-0001.md) | U-Net  | semantic segmentation       | 368x480                        |
| [yolo-v3-tiny-tf](https://github.com/openvinotoolkit/open_model_zoo/tree/develop/models/public/yolo-v3-tiny-tf)                 | YOLO v3 Tiny                          | object detection            | 416x416 |
| [yolo-v3](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/yolo-v3-tf)                               | YOLO v3                               | object detection            | 416x416 |
| [ssd-resnet34-1200-onnx](https://github.com/openvinotoolkit/open_model_zoo/tree/develop/models/public/ssd-resnet34-1200-onnx)   | ssd-resnet34 onnx model               | object detection            | 1200x1200 |

#### 7. Where can I purchase the specific hardware used in the benchmarking?
Intel partners with various vendors all over the world. Visit the [Intel® AI: In Production Partners & Solutions Catalog](https://www.intel.com/content/www/us/en/internet-of-things/ai-in-production/partners-solutions-catalog.html) for a list of Equipment Makers and the [Supported Devices](../OV_Runtime_UG/supported_plugins/Supported_Devices.md) documentation. You can also remotely test and run models before purchasing any hardware by using [Intel® DevCloud for the Edge](http://devcloud.intel.com/edge/).

#### 8. How can I optimize my models for better performance or accuracy?
We published a set of guidelines and recommendations to optimize your models available in the [optimization guide](../optimization_guide/dldt_optimization_guide.md). For further support, please join the conversation in the [Community Forum](https://software.intel.com/en-us/forums/intel-distribution-of-openvino-toolkit).

#### 9. Why are INT8 optimized models used for benchmarking on CPUs with no VNNI support?
The benefit of low-precision optimization using the OpenVINO™ toolkit model optimizer extends beyond processors supporting VNNI through Intel® DL Boost. The reduced bit width of INT8 compared to FP32 allows Intel® CPU to process the data faster and thus offers better throughput on any converted model agnostic of the intrinsically supported low-precision optimizations within Intel® hardware. Refer to [Model Accuracy for INT8 and FP32 Precision](performance_int8_vs_fp32.md) for comparison on boost factors for different network models and a selection of Intel® CPU architectures, including AVX-2 with Intel® Core™ i7-8700T, and AVX-512 (VNNI) with Intel® Xeon® 5218T and Intel® Xeon® 8270.

#### 10. Where can I search for OpenVINO™ performance results based on HW-platforms?
The web site format has changed in order to support the more common search approach of looking for the performance of a given neural network model on different HW-platforms. As opposed to review a given HW-platform's performance on different neural network models.

#### 11. How is Latency measured?
Latency is measured by running the OpenVINO™ Runtime in synchronous mode. In synchronous mode each frame or image is processed through the entire set of stages (pre-processing, inference, post-processing) before the next frame or image is processed. This KPI is relevant for applications where the inference on a single image is required, for example the analysis of an ultra sound image in a medical application or the analysis of a seismic image in the oil & gas industry. Other use cases include real-time or near real-time applications like an industrial robot's response to changes in its environment and obstacle avoidance for autonomous vehicles where a quick response to the result of the inference is required.
