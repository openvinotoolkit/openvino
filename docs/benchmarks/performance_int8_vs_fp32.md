# Model Accuracy and Performance for INT8 and FP32 {#openvino_docs_performance_int8_vs_fp32}

The following table presents the absolute accuracy drop calculated as the accuracy difference between FP32 and INT8 representations of a model:

@sphinxdirective
.. raw:: html

    <table class="table">
      <tr align="left">
        <th></th>
        <th></th>
        <th></th>
        <th>Intel® Core™ i9-12900K @ 3.2 GHz (AVX2)</th>
        <th>Intel® Xeon®  6338 @ 2.0 GHz (VNNI)</th>
        <th>iGPU Gen12LP (Intel® Core™ i9-12900K @ 3.2 GHz)</th>
      </tr>
      <tr align="left">
        <th>OpenVINO Benchmark <br>Model Name</th>
        <th>Dataset</th>
        <th>Metric Name</th>
        <th colspan="3" align="center">Absolute Accuracy Drop, %</th>
      </tr>
      <tr>
        <td>bert-base-cased</td>
        <td>SST-2</td>
        <td>accuracy</td>
        <td>0.11</td>
        <td>0.34</td>
        <td>0.46</td>
      </tr>
      <tr>
        <td>bert-large-uncased-whole-word-masking-squad-0001</td>
        <td>SQUAD</td>
        <td>F1</td>
        <td>0.87</td>
        <td>1.11</td>
        <td>0.70</td>
      </tr>      
      <tr>
        <td>deeplabv3</td>
        <td>VOC2012</td>
        <td>mean_iou</td>
        <td>0.04</td>
        <td>0.04</td>
        <td>0.11</td>
      </tr>
      <tr>
        <td>densenet-121</td>
        <td>ImageNet</td>
        <td>accuracy@top1</td>
        <td>0.56</td>
        <td>0.56</td>
        <td>0.63</td>
      </tr>
      <tr>
        <td>efficientdet-d0</td>
        <td>COCO2017</td>
        <td>coco_precision</td>
        <td>0.63</td>
        <td>0.62</td>
        <td>0.45</td>
      </tr>
      <tr>
        <td>faster_rcnn_<br>resnet50_coco</td>
        <td>COCO2017</td>
        <td>coco_<br>precision</td>
        <td>0.52</td>
        <td>0.55</td>
        <td>0.31</td>
      </tr>
      <tr>
        <td>resnet-18</td>
        <td>ImageNet</td>
        <td>acc@top-1</td>
        <td>0.16</td>
        <td>0.16</td>
        <td>0.16</td>
      </tr>
      <tr>
        <td>resnet-50</td>
        <td>ImageNet</td>
        <td>acc@top-1</td>
        <td>0.09</td>
        <td>0.09</td>
        <td>0.09</td>
      </tr>
      <tr>
        <td>resnet-50-pytorch</td>
        <td>ImageNet</td>
        <td>acc@top-1</td>
        <td>0.13</td>
        <td>0.13</td>
        <td>0.11</td>
      </tr>
      <tr>
        <td>ssd-resnet34-1200</td>
        <td>COCO2017</td>
        <td>COCO mAp</td>
        <td>0.09</td>
        <td>0.09</td>
        <td>0.13</td>
      </tr>
      <tr>
        <td>unet-camvid-onnx-0001</td>
        <td>CamVid</td>
        <td>mean_iou@mean</td>
        <td>0.56</td>
        <td>0.56</td>
        <td>0.60</td>
      </tr>
      <tr>
        <td>yolo-v3-tiny</td>
        <td>COCO2017</td>
        <td>COCO mAp</td>
        <td>0.12</td>
        <td>0.12</td>
        <td>0.17</td>
      </tr>
      <tr>
        <td>yolo_v4</td>
        <td>COCO2017</td>
        <td>COCO mAp</td>
        <td>0.52</td>
        <td>0.52</td>
        <td>0.54</td>
      </tr>
    </table>

@endsphinxdirective