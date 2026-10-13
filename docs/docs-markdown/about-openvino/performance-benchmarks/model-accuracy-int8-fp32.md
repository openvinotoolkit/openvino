---
sidebar_position: 3
sidebar_label: "Model Accuracy"
---

# Model Accuracy

The following two tables present the absolute accuracy drop calculated as the accuracy difference
between OV-accuracy and the original frame work accuracy for FP32, and the same for INT8, BF16 and
FP16 representations of a model on three platform architectures (percent point). The third table presents the GenAI model accuracies as absolute accuracy values. Please also refer to notes below
the table for more information.

- A - Intel® Core™ Ultra 9-185H (AVX2), INT8 and FP32
- B - Intel® Xeon® 6338, (VNNI), INT8 and FP32
- C - Intel® Xeon 6972P (VNNI, AMX), INT8, BF16, FP32
- D - Intel® Arc-B60, INT8 and FP16

| OpenVINO™  Model name | dataset | Metric Name | A, INT8 | B, INT8 | C, INT8 | D, INT8 |
| --- | --- | --- | --- | --- | --- | --- |
| bert-base-cased | SST-2\_bert\_cased\_padded | [spearman@cosine](mailto:spearman@cosine) | 2.57% | 2.65% | 2.95% | 2.57% |
| mobilenet-v2 | ImageNet2012 | accuracy @ top1 | -0.91% | -0.93% | -0.98% | -1.07% |
| resnet-50 | ImageNet2012 | accuracy @ top1 | -0.12% | -0.12% | -0.15% | -0.15% |
| ssd-resnet34-1200 | COCO2017\_detection\_80cl\_bkgr | map | 0.00% | 0.01% | 0.00% | 0.06% |
| yolov26n | COCO2017\_detection\_80cl\_bkgr | map | -0.53% | -0.50% | -0.47% | -0.48% |

| OpenVINO™  Model name | dataset | Metric Name | A, FP32 | B, FP32 | C, FP32 | D, FP16 |
| --- | --- | --- | --- | --- | --- | --- |
| bert-base-cased | SST-2\_bert\_cased\_padded | [spearman@cosine](mailto:spearman@cosine) | 0.00% | 2.95% | 0.00% | 0.00% |
| mobilenet-v2 | ImageNet2012 | accuracy @ top1 | 0.00% | 0.00% | -0.91% | 0.00% |
| resnet-50 | ImageNet2012 | accuracy @ top1 | 0.00% | 0.00% | -0.15% | 0.00% |
| ssd-resnet34-1200 | COCO2017\_detection\_80cl\_bkgr | map | 0.02% | 0.02% | 0.04% | 0.02% |
| yolo\_v11 | COCO2017\_detection\_80cl | [AP@0.5](mailto:AP@0.5):0.05:0.95 | -0.03% | -2.21% | -2.21% | -2.21% |
| yolo\_v26 | COCO2017\_detection\_80cl | [AP@0.5](mailto:AP@0.5):0.05:0.95 | 0.00% | 0.00% | -0.47% | 0.00% |

<table>
  <thead>
    <tr>
      <th>OpenVINO™  Model name</th>
      <th>dataset</th>
      <th>Metric Name</th>
      <th>A, AMX-FP16</th>
      <th>B, AMX-INT4</th>
      <th>C, Arc-FP16</th>
      <th>D, Arc-INT4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>DeepSeek-R1-Distill-Llama-8B</td>
      <td>Data Default WWB</td>
      <td>Similarity</td>
      <td>97.9%</td>
      <td>91.5%</td>
      <td>98.2%</td>
      <td>92.8%</td>
    </tr>
    <tr>
      <td>Gemma3n-e4b-it</td>
      <td>Data Default WWB</td>
      <td>Similarity</td>
      <td>%</td>
      <td>%</td>
      <td>%</td>
      <td>%</td>
    </tr>
    <tr>
      <td>Gemma-4-e2b-it</td>
      <td>Data Default WWB</td>
      <td>Similarity</td>
      <td>98.8%</td>
      <td>90.9%</td>
      <td>98.6%</td>
      <td>%</td>
    </tr>
    <tr>
      <td>Gemma-4-26b-a4b</td>
      <td>Data Default WWB</td>
      <td>Similarity</td>
      <td>97.4%</td>
      <td>%</td>
      <td>98.8%</td>
      <td>%</td>
    </tr>
    <tr>
      <td>GPT-OSS-20B</td>
      <td>Data Default WWB</td>
      <td>Similarity</td>
      <td>97.6%</td>
      <td>94.2%</td>
      <td>%</td>
      <td>92.8%</td>
    </tr>
    <tr>
      <td>Kokoro-82m</td>
      <td>Data Default WWB</td>
      <td>Similarity</td>
      <td>%</td>
      <td>%</td>
      <td>%</td>
      <td>%</td>
    </tr>
    <tr>
      <td>Llama-3.2-3b-instruct</td>
      <td>Data Default WWB</td>
      <td>Similarity</td>
      <td>98.8%</td>
      <td>97.7%</td>
      <td>98.4%</td>
      <td>91.2%</td>
    </tr>
    <tr>
      <td>MiniCPM-V-2.6</td>
      <td>Data Default WWB</td>
      <td>Similarity</td>
      <td>94.2%</td>
      <td>94.9%</td>
      <td>94.9%</td>
      <td>91.1%</td>
    </tr>
    <tr>
      <td>Mistral-7B-instruct</td>
      <td>Data Default WWB</td>
      <td>Similarity</td>
      <td>98.5%</td>
      <td>98.0%</td>
      <td>98.5%</td>
      <td>92.9%</td>
    </tr>
    <tr>
      <td>Phi4-mini-instruct</td>
      <td>Data Default WWB</td>
      <td>Similarity</td>
      <td>97.1%</td>
      <td>97.2%</td>
      <td>98.0%</td>
      <td>94.8%</td>
    </tr>
    <tr>
      <td>Qwen3.5-9B</td>
      <td>Data Default WWB</td>
      <td>Similarity</td>
      <td>97.3%</td>
      <td>96.3%</td>
      <td>98.2%</td>
      <td>89.3%</td>
    </tr>
    <tr>
      <td>Qwen3.6-27B</td>
      <td>Data Default WWB</td>
      <td>Similarity</td>
      <td>98.2%</td>
      <td>94.4%</td>
      <td>%</td>
      <td>94.2%</td>
    </tr>
    <tr>
      <td>Stable-Diffusion-V1-5</td>
      <td>Data Default WWB</td>
      <td>Similarity</td>
      <td>97.1%</td>
      <td>94.9%</td>
      <td>93.6%</td>
      <td>99.4%</td>
    </tr>
    <tr>
      <td>LTX-VIDEO</td>
      <td>Data Default WWB</td>
      <td>Similarity</td>
      <td>90.7%</td>
      <td>%</td>
      <td>75.9%</td>
      <td>82.1%</td>
    </tr>
  </tbody>
</table>

Notes: For all accuracy metrics a "-", (minus sign), indicates an accuracy drop.
The Similarity metric is the distance from "perfect" and as such always positive.
Similarity is cosine similarity - the dot product of two vectors divided by the product of their lengths.

<link rel="stylesheet" type="text/css" href="../../_static/css/benchmark-banner.css"/>

Results may vary. For more information, see
[F.A.Q.](performance-benchmarks-faq.md) and
[Platforms, Configurations, Methodology](index.md).
See [Legal Information](../additional-resources/terms-of-use.md).
