# INT8 vs FP32 Comparison on Select Networks and Platforms {#openvino_docs_performance_int8_vs_fp32}

The table below illustrates the speed-up factor for the performance gain by switching from an FP32 representation of an OpenVINO™ supported model to its INT8 representation. 

@sphinxdirective
.. raw:: html

    <table class="table">
      <tr align="left">
        <th></th>
        <th></th>
        <th>Intel® Core™ <br>i7-8700T</th>
        <th>Intel® Core™ <br>i7-1185G7</th>
        <th>Intel® Xeon® <br>W-1290P</th>
        <th>Intel® Xeon® <br>Platinum <br>8270</th>
      </tr>
      <tr align="left">
        <th>OpenVINO <br>benchmark <br>model name</th>
        <th>Dataset</th>
        <th colspan="4" align="center">Throughput speed-up FP16-INT8 vs FP32</th>
      </tr>
      <tr>
        <td>bert-large-<br>uncased-whole-word-<br>masking-squad-0001</td>
        <td>SQuAD</td>
        <td>1.6</td>
        <td>3.1</td>
        <td>1.5</td>
        <td>2.5</td>
      </tr>
      <tr>
        <td>brain-tumor-<br>segmentation-<br>0001-MXNET</td>
        <td>BraTS</td>
        <td>1.6</td>
        <td>2.0</td>
        <td>1.8</td>
        <td>1.8</td>
      </tr>
      <tr>
        <td>deeplabv3-TF</td>
        <td>VOC 2012<br>Segmentation</td>
        <td>1.9</td>
        <td>3.0</td>
        <td>2.8</td>
        <td>3.1</td>
      </tr>
      <tr>
        <td>densenet-121-TF</td>
        <td>ImageNet</td>
        <td>1.8</td>
        <td>3.5</td>
        <td>1.9</td>
        <td>3.8</td>
      </tr>
      <tr>
        <td>facenet-<br>20180408-<br>102900-TF</td>
        <td>LFW</td>
        <td>2.1</td>
        <td>3.6</td>
        <td>2.2</td>
        <td>3.7</td>
      </tr>
      <tr>
        <td>faster_rcnn_<br>resnet50_coco-TF</td>
        <td>MS COCO</td>
        <td>1.9</td>
        <td>3.7</td>
        <td>2.0</td>
        <td>3.4</td>
      </tr>
      <tr>
        <td>inception-v3-TF</td>
        <td>ImageNet</td>
        <td>1.9</td>
        <td>3.8</td>
        <td>2.0</td>
        <td>4.1</td>
      </tr>
      <tr>
        <td>mobilenet-<br>ssd-CF</td>
        <td>VOC2012</td>
        <td>1.6</td>
        <td>3.1</td>
        <td>1.9</td>
        <td>3.6</td>
      </tr>
      <tr>
        <td>mobilenet-v2-1.0-<br>224-TF</td>
        <td>ImageNet</td>
        <td>1.5</td>
        <td>2.4</td>
        <td>1.8</td>
        <td>3.9</td>
      </tr>
      <tr>
        <td>mobilenet-v2-<br>pytorch</td>
        <td>ImageNet</td>
        <td>1.7</td>
        <td>2.4</td>
        <td>1.9</td>
        <td>4.0</td>
      </tr>
      <tr>
        <td>resnet-18-<br>pytorch</td>
        <td>ImageNet</td>
        <td>1.9</td>
        <td>3.7</td>
        <td>2.1</td>
        <td>4.2</td>
      </tr>
      <tr>
        <td>resnet-50-<br>pytorch</td>
        <td>ImageNet</td>
        <td>1.9</td>
        <td>3.6</td>
        <td>2.0</td>
        <td>3.9</td>
      </tr>
      <tr>
        <td>resnet-50-<br>TF</td>
        <td>ImageNet</td>
        <td>1.9</td>
        <td>3.6</td>
        <td>2.0</td>
        <td>3.9</td>
      </tr>
      <tr>
        <td>squeezenet1.1-<br>CF</td>
        <td>ImageNet</td>
        <td>1.7</td>
        <td>3.2</td>
        <td>1.8</td>
        <td>3.4</td>
      </tr>
      <tr>
        <td>ssd_mobilenet_<br>v1_coco-tf</td>
        <td>VOC2012</td>
        <td>1.8</td>
        <td>3.1</td>
        <td>2.0</td>
        <td>3.6</td>
      </tr>
      <tr>
        <td>ssd300-CF</td>
        <td>MS COCO</td>
        <td>1.8</td>
        <td>4.2</td>
        <td>1.9</td>
        <td>3.9</td>
      </tr>
      <tr>
        <td>ssdlite_<br>mobilenet_<br>v2-TF</td>
        <td>MS COCO</td>
        <td>1.7</td>
        <td>2.5</td>
        <td>2.4</td>
        <td>3.5</td>
      </tr>
      <tr>
        <td>yolo_v4-TF</td>
        <td>MS COCO</td>
        <td>1.9</td>
        <td>3.6</td>
        <td>2.0</td>
        <td>3.4</td>
      </tr>
      <tr>
        <td>unet-camvid-onnx-0001</td>
        <td>MS COCO</td>
        <td>1.7</td>
        <td>3.9</td>
        <td>1.7</td>
        <td>3.7</td>
      </tr>
      <tr>
        <td>ssd-resnet34-<br>1200-onnx</td>
        <td>MS COCO</td>
        <td>1.7</td>
        <td>4.0</td>
        <td>1.7</td>
        <td>3.4</td>
      </tr>
      <tr>
        <td>googlenet-v4-tf</td>
        <td>ImageNet</td>
        <td>1.9</td>
        <td>3.9</td>
        <td>2.0</td>
        <td>4.1</td>
      </tr>
      <tr>
        <td>vgg19-caffe</td>
        <td>ImageNet</td>
        <td>1.9</td>
        <td>4.7</td>
        <td>2.0</td>
        <td>4.5</td>
      </tr>
      <tr>
        <td>yolo-v3-tiny-tf</td>
        <td>MS COCO</td>
        <td>1.7</td>
        <td>3.4</td>
        <td>1.9</td>
        <td>3.5</td>
      </tr>
    </table>

@endsphinxdirective

The following table shows the absolute accuracy drop that is calculated as the difference in accuracy between the FP32 representation of a model and its INT8 representation.

@sphinxdirective
.. raw:: html

    <table class="table">
      <tr align="left">
        <th></th>
        <th></th>
        <th></th>
        <th>Intel® Core™ <br>i9-10920X CPU<br>@ 3.50GHZ (VNNI)</th>
        <th>Intel® Core™ <br>i9-9820X CPU<br>@ 3.30GHz (AVX512)</th>
        <th>Intel® Core™ <br>i7-6700K CPU<br>@ 4.0GHz (AVX2)</th>
        <th>Intel® Core™ <br>i7-1185G7 CPU<br>@ 4.0GHz (TGL VNNI)</th>
      </tr>
      <tr align="left">
        <th>OpenVINO Benchmark <br>Model Name</th>
        <th>Dataset</th>
        <th>Metric Name</th>
        <th colspan="4" align="center">Absolute Accuracy Drop, %</th>
      </tr>
      <tr>
        <td>bert-large-uncased-whole-word-masking-squad-0001</td>
        <td>SQuAD</td>
        <td>F1</td>
        <td>0.62</td>
        <td>0.71</td>
        <td>0.62</td>
        <td>0.62</td>
      </tr>
      <tr>
        <td>brain-tumor-<br>segmentation-<br>0001-MXNET</td>
        <td>BraTS</td>
        <td>Dice-index@ <br>Mean@ <br>Overall Tumor</td>
        <td>0.08</td>
        <td>0.10</td>
        <td>0.10</td>
        <td>0.08</td>
      </tr>
      <tr>
        <td>deeplabv3-TF</td>
        <td>VOC 2012<br>Segmentation</td>
        <td>mean_iou</td>
        <td>0.09</td>
        <td>0.41</td>
        <td>0.41</td>
        <td>0.09</td>
      </tr>
      <tr>
        <td>densenet-121-TF</td>
        <td>ImageNet</td>
        <td>acc@top-1</td>
        <td>0.49</td>
        <td>0.56</td>
        <td>0.56</td>
        <td>0.49</td>
      </tr>
      <tr>
        <td>facenet-<br>20180408-<br>102900-TF</td>
        <td>LFW</td>
        <td>pairwise_<br>accuracy<br>_subsets</td>
        <td>0.05</td>
        <td>0.12</td>
        <td>0.12</td>
        <td>0.05</td>
      </tr>
      <tr>
        <td>faster_rcnn_<br>resnet50_coco-TF</td>
        <td>MS COCO</td>
        <td>coco_<br>precision</td>
        <td>0.09</td>
        <td>0.09</td>
        <td>0.09</td>
        <td>0.09</td>
      </tr>
      <tr>
        <td>inception-v3-TF</td>
        <td>ImageNet</td>
        <td>acc@top-1</td>
        <td>0.02</td>
        <td>0.01</td>
        <td>0.01</td>
        <td>0.02</td>
      </tr>
      <tr>
        <td>mobilenet-<br>ssd-CF</td>
        <td>VOC2012</td>
        <td>mAP</td>
        <td>0.06</td>
        <td>0.04</td>
        <td>0.04</td>
        <td>0.06</td>
      </tr>
      <tr>
        <td>mobilenet-v2-1.0-<br>224-TF</td>
        <td>ImageNet</td>
        <td>acc@top-1</td>
        <td>0.40</td>
        <td>0.76</td>
        <td>0.76</td>
        <td>0.40</td>
      </tr>
      <tr>
        <td>mobilenet-v2-<br>PYTORCH</td>
        <td>ImageNet</td>
        <td>acc@top-1</td>
        <td>0.36</td>
        <td>0.52</td>
        <td>0.52</td>
        <td>0.36</td>
      </tr>
      <tr>
        <td>resnet-18-<br>pytorch</td>
        <td>ImageNet</td>
        <td>acc@top-1</td>
        <td>0.25</td>
        <td>0.25</td>
        <td>0.25</td>
        <td>0.25</td>
      </tr>
      <tr>
        <td>resnet-50-<br>PYTORCH</td>
        <td>ImageNet</td>
        <td>acc@top-1</td>
        <td>0.19</td>
        <td>0.21</td>
        <td>0.21</td>
        <td>0.19</td>
      </tr>
      <tr>
        <td>resnet-50-<br>TF</td>
        <td>ImageNet</td>
        <td>acc@top-1</td>
        <td>0.11</td>
        <td>0.11</td>
        <td>0.11</td>
        <td>0.11</td>
      </tr>
      <tr>
        <td>squeezenet1.1-<br>CF</td>
        <td>ImageNet</td>
        <td>acc@top-1</td>
        <td>0.64</td>
        <td>0.66</td>
        <td>0.66</td>
        <td>0.64</td>
      </tr>
      <tr>
        <td>ssd_mobilenet_<br>v1_coco-tf</td>
        <td>VOC2012</td>
        <td>COCO mAp</td>
        <td>0.17</td>
        <td>2.96</td>
        <td>2.96</td>
        <td>0.17</td>
      </tr>
      <tr>
        <td>ssd300-CF</td>
        <td>MS COCO</td>
        <td>COCO mAp</td>
        <td>0.18</td>
        <td>3.06</td>
        <td>3.06</td>
        <td>0.18</td>
      </tr>
      <tr>
        <td>ssdlite_<br>mobilenet_<br>v2-TF</td>
        <td>MS COCO</td>
        <td>COCO mAp</td>
        <td>0.11</td>
        <td>0.43</td>
        <td>0.43</td>
        <td>0.11</td>
      </tr>
      <tr>
        <td>yolo_v4-TF</td>
        <td>MS COCO</td>
        <td>COCO mAp</td>
        <td>0.06</td>
        <td>0.03</td>
        <td>0.03</td>
        <td>0.06</td>
      </tr>
      <tr>
        <td>unet-camvid-<br>onnx-0001</td>
        <td>MS COCO</td>
        <td>COCO mAp</td>
        <td>0.29</td>
        <td>0.29</td>
        <td>0.31</td>
        <td>0.29</td>
      </tr>
      <tr>
        <td>ssd-resnet34-<br>1200-onnx</td>
        <td>MS COCO</td>
        <td>COCO mAp</td>
        <td>0.02</td>
        <td>0.03</td>
        <td>0.03</td>
        <td>0.02</td>
      </tr>
      <tr>
        <td>googlenet-v4-tf</td>
        <td>ImageNet</td>
        <td>COCO mAp</td>
        <td>0.08</td>
        <td>0.06</td>
        <td>0.06</td>
        <td>0.06</td>
      </tr>
      <tr>
        <td>vgg19-caffe</td>
        <td>ImageNet</td>
        <td>COCO mAp</td>
        <td>0.02</td>
        <td>0.04</td>
        <td>0.04</td>
        <td>0.02</td>
      </tr>
      <tr>
        <td>yolo-v3-tiny-tf</td>
        <td>MS COCO</td>
        <td>COCO mAp</td>
        <td>0.02</td>
        <td>0.6</td>
        <td>0.6</td>
        <td>0.02</td>
      </tr>
    </table>

@endsphinxdirective

![INT8 vs FP32 Comparison](../img/int8vsfp32.png)

For more complete information about performance and benchmark results, visit: [www.intel.com/benchmarks](https://www.intel.com/benchmarks) and [Optimization Notice](https://software.intel.com/articles/optimization-notice). [Legal Information](../Legal_Information.md).
