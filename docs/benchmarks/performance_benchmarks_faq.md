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
All of the performance benchmarks were generated using the open-sourced tool within the Intel® Distribution of OpenVINO™ toolkit called `benchmark_app`, which is available in both [C++](../../inference-engine/samples/benchmark_app/README.md) and [Python](../../inference-engine/tools/benchmark_tool/README.md). 

#### 6. What image sizes are used for the classification network models?
The image size used in the inference depends on the network being benchmarked. The following table shows the list of input sizes for each network model.
|   **Model**																														 |   **Public Network**                    |     **Task**                | **Input Size** (Height x Width)   |
|------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------|-----------------------------|-----------------------------------|
|    [bert-large-uncased-whole-word-masking-squad](https://github.com/openvinotoolkit/open_model_zoo/tree/develop/models/intel/bert-large-uncased-whole-word-masking-squad-int8-0001)   | 	BERT-large	|question / answer	|384|
|    [deeplabv3-TF](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/deeplabv3)                                    |	  DeepLab v3 Tf	                       |semantic segmentation	     |    513x513                          |
|    [densenet-121-TF](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/densenet-121-tf)                  | 	  Densenet-121 Tf	                   |classification	    |    224x224                 |
|    [facenet-20180408-102900-TF](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/facenet-20180408-102900)        |    FaceNet TF                           |    face recognition         |    160x160                        |
|    [faster_rcnn_resnet50_coco-TF](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/faster_rcnn_resnet50_coco)    |    Faster RCNN Tf                       |    object detection           |    600x1024					     |
|    [googlenet-v1-TF](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/googlenet-v1-tf)				     |    GoogLeNet_ILSVRC-2012                |    classification           |    224x224				  |
|    [inception-v3-TF](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/googlenet-v3)								 |    Inception v3 Tf                      |    classification           |    299x299				  |
|    [mobilenet-ssd-CF](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/mobilenet-ssd)						     |    SSD (MobileNet)_COCO-2017_Caffe      |    object detection         |    300x300						 |
|    [mobilenet-v1-1.0-224-TF](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/mobilenet-v1-1.0-224-tf)  |    MobileNet v1 Tf                      |    classification    |    224x224                        |
|    [mobilenet-v2-1.0-224-TF](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/mobilenet-v2-1.0-224)			     |    MobileNet v2 Tf                      |    classification           |    224x224						 |
|    [mobilenet-v2-pytorch](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/mobilenet-v2-pytorch )		 |    Mobilenet V2 PyTorch                 |    classification           |    224x224					     |
|    [resnet-18-pytorch](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/resnet-18-pytorch)		  			     |    ResNet-18 PyTorch                    |    classification           |    224x224						 |
|    [resnet-50-pytorch](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/resnet-50-pytorch)              |    ResNet-50 v1 PyTorch                 |    classification           |    224x224                        |
|    [resnet-50-TF](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/resnet-50-tf)								 |    ResNet-50_v1_ILSVRC-2012             |    classification           |    224x224						 |
|    [se-resnext-50-CF](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/se-resnext-50)						     |    Se-ResNext-50_ILSVRC-2012_Caffe      |    classification           |    224x224						 |
|    [squeezenet1.1-CF](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/squeezenet1.1)						     |    SqueezeNet_v1.1_ILSVRC-2012_Caffe    |    classification           |    227x227						 |
|    [ssd300-CF](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/ssd300)										     |    SSD (VGG-16)_VOC-2007_Caffe          |    object detection         |    300x300						 |
|    [yolo_v3-TF](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/yolo-v3-tf)                            | 	  TF Keras YOLO v3 Modelset            |	 object detection	      |    300x300                        |
|    [yolo_v4-TF](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/yolo-v4-tf)                            | 	  Yolo-V4 TF                           |	 object detection	     |    608x608                        |
|    [ssd_mobilenet_v1_coco-TF](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/ssd_mobilenet_v1_coco)   |    ssd_mobilenet_v1_coco                |    object detection         |    300x300                        |
|    [ssdlite_mobilenet_v2-TF](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/ssdlite_mobilenet_v2)     |    ssd_mobilenet_v2                     |    object detection         |    300x300                        |
|    [unet-camvid-onnx-0001](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/intel/unet-camvid-onnx-0001/description/unet-camvid-onnx-0001.md)            |    U-Net                    |    semantic segmentation          |    368x480                        |

#### 7. Where can I purchase the specific hardware used in the benchmarking?
Intel partners with various vendors all over the world. Visit the [Intel® AI: In Production Partners & Solutions Catalog](https://www.intel.com/content/www/us/en/internet-of-things/ai-in-production/partners-solutions-catalog.html) for a list of Equipment Makers and the [Supported Devices](../IE_DG/supported_plugins/Supported_Devices.md) documentation. You can also remotely test and run models before purchasing any hardware by using [Intel® DevCloud for the Edge](http://devcloud.intel.com/edge/).

#### 8. How can I optimize my models for better performance or accuracy?
We published a set of guidelines and recommendations to optimize your models available in an [introductory](../IE_DG/Intro_to_Performance.md) guide and an [advanced](../optimization_guide/dldt_optimization_guide.md) guide. For further support, please join the conversation in the [Community Forum](https://software.intel.com/en-us/forums/intel-distribution-of-openvino-toolkit).

#### 9. Why are INT8 optimized models used for benchmarking on CPUs with no VNNI support?
The benefit of low-precision optimization using the OpenVINO™ toolkit model optimizer extends beyond processors supporting VNNI through Intel® DL Boost. The reduced bit width of INT8 compared to FP32 allows Intel® CPU to process the data faster and thus offers better throughput on any converted model agnostic of the intrinsically supported low-precision optimizations within Intel® hardware. Please refer to [INT8 vs. FP32 Comparison on Select Networks and Platforms](performance_int8_vs_fp32.md) for comparison on boost factors for different network models and a selection of Intel® CPU architectures, including AVX-2 with Intel® Core™ i7-8700T, and AVX-512 (VNNI) with Intel® Xeon® 5218T and Intel® Xeon® 8270.

#### 10. Previous releases included benchmarks on googlenet-v1-CF (Caffe). Why is there no longer benchmarks on this neural network model?
We replaced googlenet-v1-CF to resnet-18-pytorch due to changes in developer usage. The public model resnet-18 is used by many developers as an Image Classification model. This pre-optimized model was also trained on the ImageNet database, similar to googlenet-v1-CF. Both googlenet-v1-CF and resnet-18 will remain part of the Open Model Zoo. Developers are encouraged to utilize resnet-18-pytorch for Image Classification use cases.

#### 11. Why have resnet-50-CF, mobilenet-v1-1.0-224-CF, mobilenet-v2-CF and resnet-101-CF been removed?
The CAFFE version of resnet-50, mobilenet-v1-1.0-224 and mobilenet-v2 have been replaced with their TensorFlow and PyTorch counterparts. Resnet-50-CF is replaced by resnet-50-TF, mobilenet-v1-1.0-224-CF is replaced by mobilenet-v1-1.0-224-TF and mobilenet-v2-CF is replaced by mobilenetv2-PyTorch. Resnet-50-CF an resnet-101-CF are no longer maintained at their public source repos.

#### 12. Where can I search for OpenVINO™ performance results based on HW-platforms?
The web site format has changed in order to support the more common search approach of looking for the performance of a given neural network model on different HW-platforms. As opposed to review a given HW-platform's performance on different neural network models.

#### 13. How is Latency measured?
Latency is measured by running the OpenVINO™ inference engine in synchronous mode. In synchronous mode each frame or image is processed through the entire set of stages (pre-processing, inference, post-processing) before the next frame or image is processed. This KPI is relevant for applications where the inference on a single image is required, for example the analysis of an ultra sound image in a medical application or the analysis of a seismic image in the oil & gas industry. Other use cases include real-time or near real-time applications like an industrial robot's response to changes in its environment and obstacle avoidance for autonomous vehicles where a quick response to the result of the inference is required.

\htmlonly
<style>
    .footer {
        display: none;
    }
</style>
<div class="opt-notice-wrapper">
<p class="opt-notice">
\endhtmlonly
For more complete information about performance and benchmark results, visit: [www.intel.com/benchmarks](https://www.intel.com/benchmarks) and [Optimization Notice](https://software.intel.com/articles/optimization-notice). [Legal Information](../Legal_Information.md).
\htmlonly
</p>
</div>
\endhtmlonly
