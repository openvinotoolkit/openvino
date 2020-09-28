# INT8 vs FP32 Comparison on Select Networks and Platforms {#openvino_docs_performance_int8_vs_fp32}

The table below illustrates the speed-up factor for the performance gain by switching from an FP32 representation of an OpenVINO™ supported model to its INT8 representation. 

<table>
  <tr align="left">
    <th></th>
    <th></th>
    <th>Intel® Core™ <br>i7-8700T</th>
    <th>Intel® Xeon® <br>Gold <br>5218T</th>
    <th>Intel® Xeon® <br>Platinum <br>8270</th>
    <th>Intel® Core™ <br>i7-1065G7</th>
  </tr>
  <tr align="left">
    <th>OpenVINO <br>benchmark <br>model name</th>
    <th>Dataset</th>
    <th colspan="4" align="center">Throughput speed-up FP16-INT8 vs FP32</th>
  </tr>
  <tr>
    <td>bert-large-<br>uncased-whole-word-<br>masking-squad-0001</td>
    <td>SQuAD</td>
    <td>1.5</td>
    <td>2.5</td>
    <td>2.0</td>
    <td>N/A</td>
  </tr>
  <tr>
    <td>brain-tumor-<br>segmentation-<br>0001-mxnet</td>
    <td>BraTS</td>
    <td>1.5</td>
    <td>1.7</td>
    <td>1.6</td>
    <td>1.9</td>
  </tr>
  <tr>
    <td>brain-tumor-<br>segmentation-<br>0002-cf2</td>
    <td>BraTS<br>2017</td>
    <td>1.2</td>
    <td>1.7</td>
    <td>1.4</td>
    <td>2.2</td>
  </tr>
  <tr>
    <td>deeplabv3-tf</td>
    <td>VOC 2012<br>Segmentation</td>
    <td>1.5</td>
    <td>2.4</td>
    <td>2.6</td>
    <td>2.8</td>
  </tr>
  <tr>
    <td>facenet-<br>20180408-<br>102900-tf</td>
    <td>LFW</td>
    <td>2.0</td>
    <td>3.5</td>
    <td>3.5</td>
    <td>3.5</td>
  </tr>
  <tr>
    <td>faster_rcnn_<br>resnet50_coco-tf</td>
    <td>MS COCO</td>
    <td>1.7</td>
    <td>3.4</td>
    <td>3.4</td>
    <td>3.6</td>
  </tr>
  <tr>
    <td>googlenet-v1-caffe</td>
    <td>ImageNet</td>
    <td>1.5</td>
    <td>2.9</td>
    <td>2.2</td>
    <td>3.2</td>
  </tr>
  <tr>
    <td>inception-v3-tf</td>
    <td>ImageNet</td>
    <td>1.8</td>
    <td>3.8</td>
    <td>4.0</td>
    <td>3.7</td>
  </tr>
  <tr>
    <td>mobilenet-<br>ssd-caffe</td>
    <td>VOC2012</td>
    <td>1.5</td>
    <td>3.0</td>
    <td>3.3</td>
    <td>3.1</td>
  </tr>
  <tr>
    <td>mobilenet-v1-1.0-<br>224-caffe</td>
    <td>ImageNet</td>
    <td>1.5</td>
    <td>3.2</td>
    <td>3.9</td>
    <td>2.9</td>
  </tr>
  <tr>
    <td>mobilenet-v2-1.0-<br>224-tf</td>
    <td>ImageNet</td>
    <td>1.3</td>
    <td>2.6</td>
    <td>3.8</td>
    <td>2.2</td>
  </tr>
  <tr>
    <td>mobilenet-v2-<br>caffe</td>
    <td>ImageNet</td>
    <td>1.3</td>
    <td>2.5</td>
    <td>3.4</td>
    <td>2.2</td>
  </tr>
  <tr>
    <td>resnet-101-<br>caffe</td>
    <td>ImageNet</td>
    <td>1.8</td>
    <td>3.7</td>
    <td>3.7</td>
    <td>3.6</td>
  </tr>
  <tr>
    <td>resnet-18-<br>pytorch</td>
    <td>ImageNet</td>
    <td>1.9</td>
    <td>3.7</td>
    <td>3.8</td>
    <td>3.6</td>
  </tr>
  <tr>
    <td>resnet-50-<br>caffe</td>
    <td>ImageNet</td>
    <td>1.8</td>
    <td>3.6</td>
    <td>3.9</td>
    <td>3.5</td>
  </tr>
  <tr>
    <td>resnet-50-<br>pytorch</td>
    <td>ImageNet</td>
    <td>1.8</td>
    <td>3.6</td>
    <td>3.9</td>
    <td>3.5</td>
  </tr>
  <tr>
    <td>squeezenet1.1-<br>caffe</td>
    <td>ImageNet</td>
    <td>1.6</td>
    <td>2.9</td>
    <td>3.2</td>
    <td>3.0</td>
  </tr>
  <tr>
    <td>ssd_mobilenet_<br>v1_coco-tf</td>
    <td>MS COCO</td>
    <td>1.6</td>
    <td>3.0</td>
    <td>3.4</td>
    <td>3.1</td>
  </tr>
  <tr>
    <td>ssd300-caffe</td>
    <td>MS COCO</td>
    <td>1.8</td>
    <td>3.7</td>
    <td>3.7</td>
    <td>3.8</td>
  </tr>
  <tr>
    <td>ssdlite_<br>mobilenet_<br>v2-tf</td>
    <td>MS COCO</td>
    <td>1.4</td>
    <td>2.3</td>
    <td>3.0</td>
    <td>2.4</td>
  </tr>
</table>

The following table shows the absolute accuracy drop that is calculated as the difference in accuracy between the FP32 representation of a model and its INT8 representation.

<table>
  <tr align="left">
    <th></th>
    <th></th>
    <th></th>
    <th>Intel® Core™ <br>i9-10920X CPU<br>@ 3.50GHZ (VNNI)</th>
    <th>Intel® Core™ <br>i9-9820X CPU<br>@ 3.30GHz (AVX512)</th>
    <th>Intel® Core™ <br>i7-8700 CPU<br>@ 3.20GHz (AVX2)</th>
  </tr>
  <tr align="left">
    <th>OpenVINO Benchmark <br>Model Name</th>
    <th>Dataset</th>
    <th>Metric Name</th>
    <th colspan="3" align="center">Absolute Accuracy Drop, %</th>
  </tr>
  <tr>
    <td>bert-large-<br>uncased-whole-word-<br>masking-squad-0001</td>
    <td>SQuAD</td>
    <td>F1</td>
    <td>0.46</td>
    <td>0.70</td>
    <td>0.64</td>
  </tr>
  <tr>
    <td>brain-tumor-<br>segmentation-<br>0001-mxnet</td>
    <td>BraTS</td>
    <td>Dice-index@ <br>Mean@ <br>Overall Tumor</td>
    <td>0.08</td>
    <td>0.08</td>
    <td>0.14</td>
  </tr>
  <tr>
    <td>brain-tumor-<br>segmentation-<br>0002-cf2</td>
    <td>BraTS<br>2017</td>
    <td>Dice-index@ <br>Mean@ <br>Overall Tumor</td>
    <td>0.16</td>
    <td>0.14</td>
    <td>0.13</td>
  </tr>
  <tr>
    <td>deeplabv3-tf</td>
    <td>VOC 2012<br>Segmentation</td>
    <td>mean_iou</td>
    <td>0.28</td>
    <td>0.71</td>
    <td>0.71</td>
  </tr>
  <tr>
    <td>facenet-<br>20180408-<br>102900-tf</td>
    <td>LFW</td>
    <td>pairwise_<br>accuracy<br>_subsets</td>
    <td>0.02</td>
    <td>0.05</td>
    <td>0.05</td>
  </tr>
  <tr>
    <td>faster_rcnn_<br>resnet50_coco-tf</td>
    <td>MS COCO</td>
    <td>coco_<br>precision</td>
    <td>0.21</td>
    <td>0.20</td>
    <td>0.20</td>
  </tr>
  <tr>
    <td>googlenet-v1-caffe</td>
    <td>ImageNet</td>
    <td>acc@top-1</td>
    <td>0.24</td>
    <td>0.19</td>
    <td>0.20</td>
  </tr>
  <tr>
    <td>inception-v3-tf</td>
    <td>ImageNet</td>
    <td>acc@top-1</td>
    <td>0.03</td>
    <td>0.01</td>
    <td>0.01</td>
  </tr>
  <tr>
    <td>mobilenet-<br>ssd-caffe</td>
    <td>VOC2012</td>
    <td>mAP</td>
    <td>0.35</td>
    <td>0.34</td>
    <td>0.34</td>
  </tr>
  <tr>
    <td>mobilenet-v1-1.0-<br>224-caffe</td>
    <td>ImageNet</td>
    <td>acc@top-1</td>
    <td>0.19</td>
    <td>0.18</td>
    <td>0.18</td>
  </tr>
  <tr>
    <td>mobilenet-v2-1.0-<br>224-tf</td>
    <td>ImageNet</td>
    <td>acc@top-1</td>
    <td>0.45</td>
    <td>0.94</td>
    <td>0.94</td>
  </tr>
  <tr>
    <td>mobilenet-v2-<br>caffe</td>
    <td>ImageNet</td>
    <td>acc@top-1</td>
    <td>0.24</td>
    <td>1.45</td>
    <td>1.45</td>
  </tr>
  <tr>
    <td>resnet-101-<br>caffe</td>
    <td>ImageNet</td>
    <td>acc@top-1</td>
    <td>0.00</td>
    <td>0.02</td>
    <td>0.02</td>
  </tr>
  <tr>
    <td>resnet-18-<br>pytorch</td>
    <td>ImageNet</td>
    <td>acc@top-1</td>
    <td>0.26</td>
    <td>0.25</td>
    <td>0.25</td>
  </tr>
  <tr>
    <td>resnet-50-<br>caffe</td>
    <td>ImageNet</td>
    <td>acc@top-1</td>
    <td>0.16</td>
    <td>0.12</td>
    <td>0.12</td>
  </tr>
  <tr>
    <td>resnet-50-<br>pytorch</td>
    <td>ImageNet</td>
    <td>acc@top-1</td>
    <td>0.20</td>
    <td>0.17</td>
    <td>0.17</td>
  </tr>
  <tr>
    <td>squeezenet1.1-<br>caffe</td>
    <td>ImageNet</td>
    <td>acc@top-1</td>
    <td>0.66</td>
    <td>0.64</td>
    <td>0.64</td>
  </tr>
  <tr>
    <td>ssd_mobilenet_<br>v1_coco-tf</td>
    <td>MS COCO</td>
    <td>COCO mAp</td>
    <td>0.24</td>
    <td>3.07</td>
    <td>3.07</td>
  </tr>
  <tr>
    <td>ssd300-caffe</td>
    <td>MS COCO</td>
    <td>COCO mAp</td>
    <td>0.06</td>
    <td>0.05</td>
    <td>0.05</td>
  </tr>
  <tr>
    <td>ssdlite_<br>mobilenet_<br>v2-tf</td>
    <td>MS COCO</td>
    <td>COCO mAp</td>
    <td>0.14</td>
    <td>0.47</td>
    <td>0.47</td>
  </tr>
</table>

![INT8 vs FP32 Comparison](img/int8vsfp32.png "INT8 vs FP32 Comparison on Select Networks and Platforms")

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