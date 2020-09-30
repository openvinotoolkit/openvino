# Performance Information Frequently Asked Questions {#openvino_docs_performance_benchmarks_faq}

The following questions and answers are related to [performance benchmarks](./performance_benchmarks.md) published on the documentation site.

#### 1. How often do performance benchmarks get updated?
New performance benchmarks are typically published on every `major.minor` release of the Intel® Distribution of OpenVINO™ toolkit.

#### 2. Where can I find the models used in the performance benchmarks?
All of the models used are included in the toolkit's [Open Model Zoo](https://github.com/opencv/open_model_zoo) GitHub repository. 

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
|    [faster_rcnn_resnet50_coco-TF](https://github.com/opencv/open_model_zoo/tree/master/models/public/faster_rcnn_resnet50_coco)    |    Faster RCNN Tf                       |    object detection         |    600x1024					     |
|    [googlenet-v1-CF](https://github.com/opencv/open_model_zoo/tree/master/models/public/googlenet-v1)							     |    GoogLeNet_ILSVRC-2012_Caffe          |    classification           |    224x224				         |
|    [googlenet-v3-TF](https://github.com/opencv/open_model_zoo/tree/master/models/public/googlenet-v3)								 |    Inception v3 Tf                      |    classification           |    299x299				   	     |
|    [mobilenet-ssd-CF](https://github.com/opencv/open_model_zoo/tree/master/models/public/mobilenet-ssd)						     |    SSD   (MobileNet)_COCO-2017_Caffe    |    object detection         |    300x300						 |
|    [mobilenet-v2-1.0-224-TF](https://github.com/opencv/open_model_zoo/tree/master/models/public/mobilenet-v2-1.0-224)			     |    MobileNet v2 Tf                      |    classification           |    224x224						 |
|    [mobilenet-v2-CF](https://github.com/opencv/open_model_zoo/tree/master/models/public/mobilenet-v2)								 |    Mobilenet V2 Caffe                   |    classification           |    224x224					     |
|    [resnet-101-CF](https://github.com/opencv/open_model_zoo/tree/master/models/public/resnet-101)									 |    ResNet-101_ILSVRC-2012_Caffe         |    classification           |    224x224						 |
|    [resnet-50-CF](https://github.com/opencv/open_model_zoo/tree/master/models/public/resnet-50)									 |    ResNet-50_v1_ILSVRC-2012_Caffe       |    classification           |    224x224						 |
|    [se-resnext-50-CF](https://github.com/opencv/open_model_zoo/tree/master/models/public/se-resnext-50)						     |    Se-ResNext-50_ILSVRC-2012_Caffe      |    classification           |    224x224						 |
|    [squeezenet1.1-CF](https://github.com/opencv/open_model_zoo/tree/master/models/public/squeezenet1.1)						     |    SqueezeNet_v1.1_ILSVRC-2012_Caffe    |    classification           |    227x227						 |
|    [ssd300-CF](https://github.com/opencv/open_model_zoo/tree/master/models/public/ssd300)										     |    SSD (VGG-16)_VOC-2007_Caffe          |    object detection         |    300x300						 |

#### 7. Where can I purchase the specific hardware used in the benchmarking?
Intel partners with various vendors all over the world. Visit the [Intel® AI: In Production Partners & Solutions Catalog](https://www.intel.com/content/www/us/en/internet-of-things/ai-in-production/partners-solutions-catalog.html) for a list of Equipment Makers and the [Supported Devices](../IE_DG/supported_plugins/Supported_Devices.md) documentation. You can also remotely test and run models before purchasing any hardware by using [Intel® DevCloud for the Edge](http://devcloud.intel.com/edge/).

#### 8. How can I optimize my models for better performance or accuracy?
We published a set of guidelines and recommendations to optimize your models available in an [introductory](../IE_DG/Intro_to_Performance.md) guide and an [advanced](../optimization_guide/dldt_optimization_guide.md) guide. For further support, please join the conversation in the [Community Forum](https://software.intel.com/en-us/forums/intel-distribution-of-openvino-toolkit).

#### 9. Why are INT8 optimized models used for benchmarking on CPUs with no VNNI support?
The benefit of low-precision optimization using the OpenVINO™ toolkit model optimizer extends beyond processors supporting VNNI through Intel® DL Boost. The reduced bit width of INT8 compared to FP32 allows Intel® CPU to process the data faster and thus offers better throughput on any converted model agnostic of the intrinsically supported low-precision optimizations within Intel® hardware. Please refer to [INT8 vs. FP32 Comparison on Select Networks and Platforms](./performance_int8_vs_fp32.html) for comparison on boost factors for different network models and a selection of Intel® CPU architectures, including AVX-2 with Intel® Core™ i7-8700T, and AVX-512 (VNNI) with Intel® Xeon® 5218T and Intel® Xeon® 8270.

#### 10. Previous releases included benchmarks on googlenet-v1. Why is there no longer benchmarks on this neural network model?
We replaced googlenet-v1 to [resnet-18-pytorch](https://github.com/opencv/open_model_zoo/blob/master/models/public/resnet-18-pytorch/resnet-18-pytorch.md) due to changes in developer usage. The public model resnet-18 is used by many developers as an Image Classification model. This pre-optimized model was also trained on the ImageNet database, similar to googlenet-v1. Both googlenet-v1 and resnet-18 will remain part of the Open Model Zoo. Developers are encouraged to utilize resnet-18-pytorch for Image Classification use cases.


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
