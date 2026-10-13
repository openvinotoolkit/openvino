---
sidebar_position: 3
sidebar_label: "AI Models verified for OpenVINO™"
---

# AI Models verified for OpenVINO™

The following is a list of models that have been verified to work with OpenVINO. Note that other
models from OpenVINO-supported frameworks may also work properly but have not been tested.

**AI Models that run on Intel® Core Ultra™ Processors with OpenVINO™ toolkit:**

<table>
  <thead>
    <tr>
      <th>Category</th>
      <th>Topology</th>
      <th>Source Framework</th>
      <th>Precision</th>
      <th>Inferred With</th>
      <th>CPU</th>
      <th>GPU</th>
      <th>NPU</th>
      <th>Last Verified</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Sound Classification</td>
      <td>aclnet</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Sound Classification</td>
      <td>aclnet</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Sound Classification</td>
      <td>aclnet</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Sound Classification</td>
      <td>aclnet-int8</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>action-recognition-0001</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>action-recognition-0001</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>afm-4.5b</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>afm-4.5b</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>afm-4.5b</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>Albert</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>albert-base-v2</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>albert-base-v2</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>albert-base-v2</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>albert-base-v2</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>albert-base-v2-sst2</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>albert-large-v2</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>albert-large-v2</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>albert-large-v2-finetuned-mnli</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>alexnet</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>alexnet</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>alexnet</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>alexnet</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>alexnet</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>alexnet</td>
      <td>pytorch</td>
      <td>FP32-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>anti-spoof-mn3</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>anti-spoof-mn3</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>anti-spoof-mn3</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>Aquila2-7B</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>Aquila-7B</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>AquilaChat2-7B</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>AquilaChat-7B</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>asl-recognition-0003</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>asl-recognition-0003</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>asl-recognition-0003</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>asl-recognition-0004</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>asl-recognition-0004</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Audio Classification</td>
      <td>ast-finetuned-speech-commands-v2</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Text Recognition</td>
      <td>attention-ocr</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Text Recognition</td>
      <td>attention-ocr</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Text Recognition</td>
      <td>attention-ocr</td>
      <td>tf</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Video Processing </td>
      <td>Background_Matting</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Semantic Segmentation</td>
      <td>background-matting-mobilenetv2</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Semantic Segmentation</td>
      <td>background-matting-mobilenetv2</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Semantic Segmentation</td>
      <td>background-matting-mobilenetv2</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>baichuan2-13b-chat</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>baichuan2-13b-chat</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>baichuan2-13b-chat</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>baichuan2-7b-chat</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>baichuan2-7b-chat</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>baichuan2-7b-chat</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>Baichuan2-7B-Chat</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>bart-base</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>bart-base-japanese</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>barthez-orangesum-abstract</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>bce-embedding-base_v1</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>bce-embedding-base_v1</td>
      <td>pytorch</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>began</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>began</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>began</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>beit_large_patch16_224.in22k_ft_in22k_in1k</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>beit-base-patch16-224-pt22k-ft22k</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>beitv2_large_patch16_224.in1k_ft_in22k_in1k</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>Bert</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>Bert_large</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>BERT_pytorch</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>bert-base-arabertv02</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>bert-base-cased</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>bert-base-cased</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>bert-base-cased</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>bert-base-cased</td>
      <td>paddle</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>bert-base-cased</td>
      <td>paddle</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>bert-base-cased</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>bert-base-cased</td>
      <td>paddle</td>
      <td>INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>bert-base-cased-conversational</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>bert-base-cased-squad2</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>bert-base-cased-squad2</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>bert-base-cased-squad2</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>bert-base-chinese-xnli-zh</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>bert-base-chinese-xnli-zh</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>bert-base-chinese-xnli-zh</td>
      <td>tf</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>bert-base-multilingual-cased</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>bert-base-multilingual-cased</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>bert-base-multilingual-cased</td>
      <td>pytorch</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>bert-base-multilingual-cased</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>bert-base-ner</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>bert-base-ner</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>bert-base-ner</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>bert-base-NER</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>bert-base-NER</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>bert-base-uncased</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>bert-base-uncased</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>bert-base-uncased-mrpc</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>bert-base-uncased-mrpc</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>bert-base-uncased-mrpc</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>bert-base-uncased-mrpc</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>bert-base-uncased-mrpc</td>
      <td>pytorch</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>bert-base-uncased-mrpc</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>bert-base-uncased-sparse-70-unstructured</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>bert-base-uncased-yelp-polarity</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>bert-large-cased</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>bert-large-cased</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>bert-large-cased</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>bert-large-cased</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>BERT-Large-CT-STSb</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>bert-large-NER</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>bert-large-uncased</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>bert-large-uncased</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>bert-large-uncased</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>bert-large-uncased</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>bert-large-uncased</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>bert-large-uncased-whole-word-masking</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>bert-large-uncased-whole-word-masking-finetuned-squad</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>bert-large-uncased-whole-word-masking-finetuned-sst-2</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>bert-large-uncased-whole-word-masking-squad-0001</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>bert-large-uncased-whole-word-masking-squad-0001</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>bert-large-uncased-whole-word-masking-squad-emb-0001</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>bert-large-uncased-whole-word-masking-squad-emb-0001</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>bert-large-uncased-whole-word-masking-squad-int8-0001</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>bert-small-finetuned-squadv2</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>bert-small-uncased-whole-word-masking-squad-0001</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>bert-small-uncased-whole-word-masking-squad-0001</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>bert-small-uncased-whole-word-masking-squad-0001</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>bert-small-uncased-whole-word-masking-squad-0002</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>bert-small-uncased-whole-word-masking-squad-0002</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>bert-small-uncased-whole-word-masking-squad-emb-int8-0001</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>bert-small-uncased-whole-word-masking-squad-int8-0002</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>bge-base-en-v1.5</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>bge-base-en-v1.5</td>
      <td>pytorch</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>bge-large-en-v1.5</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>bge-reranker-base</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>bge-reranker-large</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>BigBird</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>bigbird-pegasus-large-arxiv</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>bigbird-roberta-base</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>Bio_ClinicalBERT</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>BioMedLM</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>biomistral-7b-slerp</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>biomistral-7b-slerp</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>biomistral-7b-slerp</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>blenderbot-400M-distill</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>bloom</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>bloom-560m</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>bloomz-1b1</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>bloomz-1b4-zh</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>bloomz-1b4-zh</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>bloomz-1b4-zh</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>bloomz-3b</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>bloomz-7b1</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>bloomz-7b1</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>bloomz-7b1</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>bloomz-7b1-mt</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>botnet26t_256</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>brain-tumor-segmentation-0002</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>brain-tumor-segmentation-0002</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>brain-tumor-segmentation-0002</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>cait_s24_224</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>camembert-base</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>canine-s</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>cascade_mask_rcnn_R_50_FPN_1x</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>cascade_rcnn_resnet101_fpn_1x_coco</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>cascade_rcnn_resnet101_fpn_1x_coco</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>cascade_rcnn_resnet101_fpn_1x_coco</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>centernet-hg104</td>
      <td>tf2</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>centernet-hg104</td>
      <td>tf2</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>centernet-hg104</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>centernet-hg104</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>centernet-resnet</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>centernet-resnet</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>chatglm3-6b</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>chatglm3-6b</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>chatglm3-6b</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>chinese_pretrain_mrc_macbert_large</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>chinese-xlnet-base</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>clip-vit-b-16-plus-240-laion400m-e32</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>clip-vit-b-16-plus-240-laion400m-e32</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>clip-vit-b-32-laion2b-s34b-b79k</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>clip-vit-b-32-laion2b-s34b-b79k</td>
      <td>pytorch</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>clip-vit-base-patch16</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>clip-vit-base-patch16</td>
      <td>pytorch</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>clip-vit-base-patch32</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>clip-vit-base-patch32</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>clip-vit-large-patch14</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>clip-vit-large-patch14</td>
      <td>pytorch</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>coat_mini</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>coatnet_1_rw_224.sw_in1k</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>coatnext_nano_rw_224.sw_in1k</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>codegemma-1.1-2b</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>codegemma-1.1-7b-it</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>codegemma-2b</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>codegemma-7b</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>codegen2-1B_P</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>codegen-2B-multi</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>codegen-350M-mono</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>codellama-7b</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>codellama-7b</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>codellama-7b</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>CodeQwen1.5-7B-Chat</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>codet5-base-sum</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>codet5-base-sum</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>codet5-base-sum</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Colorization</td>
      <td>colorization-siggraph</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Colorization</td>
      <td>colorization-siggraph</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Colorization</td>
      <td>colorization-siggraph</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Colorization</td>
      <td>colorization-v2</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Colorization</td>
      <td>colorization-v2</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Colorization</td>
      <td>colorization-v2</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>common-sign-language-0001</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>common-sign-language-0001</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>common-sign-language-0002</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>common-sign-language-0002</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>common-sign-language-0002</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>contriever</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>convbert-base-turkish-mc4-toxicity-uncased</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>convit_small</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>convmixer_1024_20_ks9_p14</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>convnext_base</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>convnext_large</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>convnext_large.fb_in1k</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>convnext_small</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>convnext_tiny</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>convnext-base-224</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>convnext-tiny</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>convnext-tiny</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>convnext-tiny</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>convnext-tiny</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>convnext-tiny</td>
      <td>pytorch</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>convnext-tiny</td>
      <td>pytorch</td>
      <td>FP32-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>convnextv2_large.fcmae_ft_in22k_in1k</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>convnextv2-tiny-1k-224</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>corel_nnart_001</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>corel_nnart_001</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>corel_nnart_001</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Text Recognition</td>
      <td>crnn-tf</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Text Recognition</td>
      <td>crnn-tf</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Text Recognition</td>
      <td>crnn-tf</td>
      <td>tf</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>crossvit_15_dagger_240</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>crossvit_small_240</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>cs3darknet_focus_m</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>cs3darknet_m</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>cs3edgenet_x</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>cs3se_edgenet_x</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>cs3sedarknet_l</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>cspdarknet53</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>cspresnet50</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>cspresnext50</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>ctdet_coco_dlav0_512</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>ctdet_coco_dlav0_512</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>ctdet_coco_dlav0_512</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Text Detection</td>
      <td>ctpn</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Text Detection</td>
      <td>ctpn</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Text Detection</td>
      <td>ctpn</td>
      <td>tf</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>darknet19</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>darknet19</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>darknet19</td>
      <td>tf</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>darknet53</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Audio</td>
      <td>data2vec2-base</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Audio</td>
      <td>data2vec2-base</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>davit_small.msft_in1k</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Generation</td>
      <td>dcgan</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>deberta-base</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>deberta-large</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>deberta-v2-xlarge</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>deberta-v3-large</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>deblurgan-v2</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>deblurgan-v2</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Semantic Segmentation</td>
      <td>deeplabv3</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Semantic Segmentation</td>
      <td>deeplabv3</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Semantic Segmentation</td>
      <td>deeplabv3</td>
      <td>tf</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Semantic Segmentation</td>
      <td>deeplabv3_mobilenet_v3_large</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Semantic Segmentation</td>
      <td>deeplabv3_resnet101</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Semantic Segmentation</td>
      <td>deeplabv3_resnet50</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Segmentation</td>
      <td>deeplabv3-mobilenetv3</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Segmentation</td>
      <td>deeplabv3-mobilenetv3</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Segmentation</td>
      <td>deeplabv3-mobilenetv3</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Semantic Segmentation</td>
      <td>deeplab-v3p-resnet50-os8</td>
      <td>paddle</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Semantic Segmentation</td>
      <td>deeplab-v3p-resnet50-os8</td>
      <td>paddle</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Semantic Segmentation</td>
      <td>deeplab-v3p-resnet50-os8</td>
      <td>paddle</td>
      <td>INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>deepseek-r1-distill-llama-8b</td>
      <td>pytorch</td>
      <td>FP4-NORMALIZED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>deepseek-r1-distill-llama-8b</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>deepseek-r1-distill-llama-8b</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>deepseek-r1-distill-llama-8b</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>deepseek-r1-distill-qwen-1.5b</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>deepseek-r1-distill-qwen-1.5b</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>deepseek-r1-distill-qwen-1.5b</td>
      <td>pytorch</td>
      <td>FP4-NORMALIZED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>deepseek-r1-distill-qwen-1.5b</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>deepseek-r1-distill-qwen-14b</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>deepseek-r1-distill-qwen-14b</td>
      <td>pytorch</td>
      <td>FP4-NORMALIZED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>deepseek-r1-distill-qwen-14b</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>deepseek-r1-distill-qwen-14b</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>deepseek-r1-distill-qwen-7b</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>deepseek-r1-distill-qwen-7b</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>deepseek-r1-distill-qwen-7b</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>deepseek-r1-distill-qwen-7b</td>
      <td>pytorch</td>
      <td>FP4-NORMALIZED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>DNA Classification</td>
      <td>deepvariant_wgs</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>DNA Classification</td>
      <td>deepvariant_wgs</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>DNA Classification</td>
      <td>deepvariant_wgs</td>
      <td>tf</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>deit_small_patch16_224</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>deit3_small_patch16_224_in21ft1k</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>deit-b</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>deit-b</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>deit-b</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Audio Enhancement</td>
      <td>demucs</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Audio Enhancement</td>
      <td>Denoise</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Audio Enhancement</td>
      <td>Denoise</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Audio Enhancement</td>
      <td>Denoise</td>
      <td>tf</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>densenet121</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>densenet-121</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>densenet-121</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>densenet-121</td>
      <td>tf2</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>densenet-121</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>densenet-121</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>densenet-121</td>
      <td>tf2</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>densenet-121</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>densenet161</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>densenet-161</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>densenet-161</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>densenet-161</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>densenet-161</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>densenet-161</td>
      <td>pytorch</td>
      <td>FP32-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>densenet169</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>densenet201</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>densenetblur121d</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>detr_resnet50</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>detr_resnet50</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>detr_resnet50</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>detr-resnet-50</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Click-Through Rate (CTR) Prediction</td>
      <td>dien</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Click-Through Rate (CTR) Prediction</td>
      <td>dien</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Behavior / Decision Prediction</td>
      <td>dien_alibaba</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Behavior / Decision Prediction</td>
      <td>dien_alibaba</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>DistilBert</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>distilbert-base-cased</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>distilbert-base-cased</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>distilbert-base-cased</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>distilbert-base-nli-mean-tokens</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>distilbert-base-uncased</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>distilbert-base-uncased</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>distilbert-base-uncased</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>distilbert-base-uncased</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>distilbert-base-uncased</td>
      <td>pytorch</td>
      <td>FP32-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>distilbert-base-uncased</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>distilbert-base-uncased-distilled-squad</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>distilbert-base-uncased-distilled-squad</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>distilbert-base-uncased-finetuned-sst-2-english</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>distilbert-base-uncased-finetuned-sst-2-english</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>distilbert-base-uncased-sst-2</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>distilbert-base-uncased-sst-2</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>distilbert-base-uncased-sst-2</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>distilbert-NER</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Speech Recognition</td>
      <td>distil-large-v2</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Speech Recognition</td>
      <td>distil-large-v2</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Speech Recognition</td>
      <td>distil-large-v2</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Speech Recognition</td>
      <td>distil-large-v2-genai</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Speech Recognition</td>
      <td>distil-large-v2-genai</td>
      <td>pytorch</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Speech Recognition</td>
      <td>distil-large-v3</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Speech Recognition</td>
      <td>distil-medium.en</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>dla34</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>dla-34</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>dla-34</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>dla-34</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>dla-34</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>dla-34</td>
      <td>pytorch</td>
      <td>FP32-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>dla60_res2net</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>dla60_res2next</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>dla60x</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>DLRM</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>DLRM</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>dm_nfnet_f6</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>dm_nfnet_f6.dm_in1k</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>DNA Classification</td>
      <td>dna_r9.4.1</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>DNA Classification</td>
      <td>dna_r9.4.1</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>DNA Classification</td>
      <td>dna_r9.4.1</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>DNA Classification</td>
      <td>dna_r9.4.1_2d</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>DNA Classification</td>
      <td>dna_r9.4.1_2d</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>DNA Classification</td>
      <td>dna_r9.4.1_2d</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>dolly-v2-12b</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>dolly-v2-12b</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>dolly-v2-12b</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>dolly-v2-3b</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>dolly-v2-3b</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>dolly-v2-3b</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>dpn68</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification, Dual Path Network</td>
      <td>dpn-68</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification, Dual Path Network</td>
      <td>dpn-68</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification, Dual Path Network</td>
      <td>dpn-68</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification, Dual Path Network</td>
      <td>dpn-68</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification, Dual Path Network</td>
      <td>dpn-68</td>
      <td>pytorch</td>
      <td>FP32-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>driver-action-recognition-adas-0002</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>driver-action-recognition-adas-0002</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Semantic Segmentation</td>
      <td>drn-d-38</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Semantic Segmentation</td>
      <td>drn-d-38</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Semantic Segmentation</td>
      <td>drn-d-38</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>east_resnet_v1_50</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>east_resnet_v1_50</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>east_resnet_v1_50</td>
      <td>tf</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>ebgan</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>ebgan</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>ebgan</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>ebgan</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>ebgan</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>eca_botnext26ts_256</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>eca_halonext26ts</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>eca_nfnet_l1</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>ecaresnet50d</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>edgenext_small_rw</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>edsr3_super_resolution</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>edsr3_super_resolution</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>edsr3_super_resolution</td>
      <td>tf</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>edsr3-nas</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>edsr3-nas</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>edsr3-nas</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>efficientdet-d0</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>efficientdet-d0</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>efficientdet-d0</td>
      <td>tf</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>efficientdet-d1</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>efficientdet-d1</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>efficientformer_l3.snap_dist_in1k</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>efficientformerv2_s1.snap_dist_in1k</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>efficientnet_b0</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>efficientnet_b1</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>efficientnet_b2</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>efficientnet_b3</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>efficientnet_b4</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>efficientnet_b5</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>efficientnet_b6</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>efficientnet_b7</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>efficientnet_v2_l</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>efficientnet_v2_m</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>efficientnet_v2_s</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>efficientnet-b0</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>efficientnet-b0</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>efficientnet-b0</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>efficientnet-b0</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>efficientnet-b0</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>efficientnet-b0</td>
      <td>tf</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>efficientnet-b0_auto_aug</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>efficientnet-b0_auto_aug</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>efficientnet-b0_auto_aug</td>
      <td>tf</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>efficientnet-b3</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>efficientnet-b3</td>
      <td>pytorch</td>
      <td>FP32-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>efficientnetv2_rw_s.ra2_in1k</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>efficientnet-v2-m</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>efficientnet-v2-m</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>efficientnet-v2-m</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>efficientnet-v2-m</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>efficientnet-v2-m</td>
      <td>pytorch</td>
      <td>FP32-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>electra-base-french-europeana-cased-generator</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>electra-base-generator</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>electra-base-squad2</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>electra-large-discriminator-nli-efl-tweeteval</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>electra-small-discriminator</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>ens_adv_inception_resnet_v2</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>ens3-adv-inception-v3</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>ens3-adv-inception-v3</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>ens3-adv-inception-v3</td>
      <td>tf</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>ese_vovnet39b</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>esrgan</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>esrgan</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>esrgan</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>eva_large_patch14_336.in22k_ft_in1k</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Segmentation</td>
      <td>facebookresearch_sam2_base</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Segmentation</td>
      <td>facebookresearch_sam2_large</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Segmentation</td>
      <td>facebookresearch_sam2_small</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Segmentation</td>
      <td>facebookresearch_sam2_tiny</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>faceboxes</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>faceboxes</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>faceboxes</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>face-detection-0200</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>face-detection-0200</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>face-detection-0202</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>face-detection-0202</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>face-detection-0204</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>face-detection-0204</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>face-detection-0205</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>face-detection-0205</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>face-detection-0206</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>face-detection-0206</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>face-detection-0206</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>face-detection-retail-0005</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>face-detection-retail-0005</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>face-detection-retail-0005</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>facenet</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>facenet</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>facenet</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>facenet-20180408-102900</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>facenet-20180408-102900</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>face-recognition-resnet100-arcface</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>face-recognition-resnet100-arcface</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>face-recognition-resnet100-arcface</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>face-reidentification-retail-0095</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>face-reidentification-retail-0095</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>face-reidentification-retail-0095</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>facial-landmarks-98-detection-0001</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>facial-landmarks-98-detection-0001</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>fairseq_s2t_covost</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>falcon-11B</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>falcon-40b</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>falcon-7b</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>falcon-7b-instruct</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>falcon-7b-instruct</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>falcon-7b-instruct</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>falcon-7b-instruct</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>fara-7b</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>fara-7b</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>fara-7b</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>faster_rcnn_inception_resnet_v2_atrous_coco</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>faster_rcnn_inception_v2_coco</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>faster_rcnn_inception_v2_coco</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>faster_rcnn_R_50_C4</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>faster_rcnn_R_50_DC5_1x</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>faster_rcnn_R_50_FPN_1x</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>faster_rcnn_R_50_FPN_noaug_1x</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>faster_rcnn_resnet101_coco</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>faster_rcnn_resnet101_kitti</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>faster_rcnn_resnet50_coco</td>
      <td>tf2</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>faster_rcnn_resnet50_coco</td>
      <td>tf2</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>faster_rcnn_resnet50_coco</td>
      <td>tf2</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>faster_rcnn_resnet50_coco</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>faster_rcnn_resnet50_coco</td>
      <td>tf</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>faster_rcnn_resnet50_fpn_coco</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>faster_rcnn_resnet50_fpn_coco</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>faster_rcnn_resnet50_fpn_coco</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>faster_rcnn_resnet50_lowproposals_coco</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>faster_rcnn_X_101_32x8d_FPN_3x</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>fasterrcnn_mobilenet_v3_large_320_fpn</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>fasterrcnn_mobilenet_v3_large_fpn</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>fasterrcnn_resnet50_fpn</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image to Image</td>
      <td>fast-neural-style-mosaic</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image to Image</td>
      <td>fast-neural-style-mosaic</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image to Image</td>
      <td>fast-neural-style-mosaic</td>
      <td>onnx</td>
      <td>INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Semantic Segmentation</td>
      <td>fastseg-large</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Semantic Segmentation</td>
      <td>fastseg-large</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Semantic Segmentation</td>
      <td>fastseg-small</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Semantic Segmentation</td>
      <td>fastseg-small</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Semantic Segmentation</td>
      <td>fastseg-small</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>fbcnn</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>fbcnn</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>fbcnn</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>fbnetc_100.rmsp_in1k</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>fbnetc-100</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>fbnetc-100</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>fbnetc-100</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>fbnetc-100</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>fbnetc-100</td>
      <td>pytorch</td>
      <td>FP32-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>fbnetv3_d.ra2_in1k</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Segmentation</td>
      <td>fcn_resnet101</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Segmentation</td>
      <td>fcn_resnet50</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>fcos_resnet50_fpn</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Monodepth</td>
      <td>fcrn-dp-nyu-depth-v2-tf</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Monodepth</td>
      <td>fcrn-dp-nyu-depth-v2-tf</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>flan-t5-large-grammar-synthesis</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>flan-t5-large-grammar-synthesis</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>flan-t5-large-grammar-synthesis</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>flan-t5-large-grammar-synthesis</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>flan-t5-xxl</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>flan-t5-xxl</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>flan-t5-xxl</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>flexivit_base.300ep_in21k</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>flexivit_base.patch16_in21k</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Generation</td>
      <td>flux.1-dev</td>
      <td>pytorch</td>
      <td>INT8-HYBRID</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Generation</td>
      <td>flux.1-dev</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Generation</td>
      <td>flux.1-schnell</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Generation</td>
      <td>flux.1-schnell</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Generation</td>
      <td>flux.1-schnell</td>
      <td>pytorch</td>
      <td>INT8-HYBRID</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>fnet-base</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Text to Speech</td>
      <td>forward-tacotron-duration-prediction</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Text to Speech</td>
      <td>forward-tacotron-duration-prediction</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Text to Speech</td>
      <td>forward-tacotron-duration-prediction</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Text to Speech</td>
      <td>forward-tacotron-regression</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Text to Speech</td>
      <td>forward-tacotron-regression</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Text to Speech</td>
      <td>forward-tacotron-regression</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>fsrcnn-x4</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>fsrcnn-x4</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>fsrcnn-x4</td>
      <td>tf</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Text Recognition</td>
      <td>functorch_maml_omniglot</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>gaze-estimation-adas-0002</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>gaze-estimation-adas-0002</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>gaze-estimation-adas-0002</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>gc_efficientnetv2_rw_t.agc_in1k</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>gcresnet50t</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>gcvit_tiny</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>gelectra-base-germanquad</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>gemma-1.1-2b-it</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>gemma-1.1-7b-it</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>gemma-2-2b</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>gemma-2-2b</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>gemma-2-2b</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>gemma-2-9b-it</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>gemma-2-9b-it</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>gemma-2-9b-it</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>gemma-2b</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>gemma-2b-it</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>gemma-2b-it</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>gemma-2b-it</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>gemma-2b-it</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Multimodal Large Language Model</td>
      <td>gemma-3-12b-it</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Multimodal Large Language Model</td>
      <td>gemma-3-12b-it</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Multimodal Large Language Model</td>
      <td>gemma-3-12b-it</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Multimodal Large Language Model</td>
      <td>gemma-3-1b-it</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Multimodal Large Language Model</td>
      <td>gemma-3-1b-it</td>
      <td>pytorch</td>
      <td>FP4-NORMALIZED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Multimodal Large Language Model</td>
      <td>gemma-3-1b-it</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Multimodal Large Language Model</td>
      <td>gemma-3-1b-it</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>gemma-3-270m</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>gemma-3-270m</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>gemma-3-270m</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Multimodal Large Language Model</td>
      <td>gemma-3-4b-it</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Multimodal Large Language Model</td>
      <td>gemma-3-4b-it</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Multimodal Large Language Model</td>
      <td>gemma-3-4b-it</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Multimodal Large Language Model</td>
      <td>gemma-3-4b-it</td>
      <td>pytorch</td>
      <td>FP4-NORMALIZED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>gemma-4-26b-a4b-it</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>gemma-4-31b-it</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>gemma-4-e2b-it</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>gemma-4-e2b-it</td>
      <td>pytorch</td>
      <td>FP4-NORMALIZED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>gemma-4-e2b-it</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>gemma-4-e2b-it</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>gemma-7b</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>gemma-7b-it</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>gemma-7b-it</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>gemma-7b-it</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>gemma-7b-it</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>gernet_m</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>ghostnet_100</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>glm-4-9b-chat-hf</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>glm-4-9b-chat-hf</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>glm-4-9b-chat-hf</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>glm-edge-1.5b-chat</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>glm-edge-1.5b-chat</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>glm-edge-1.5b-chat</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>glm-edge-4b-chat</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>glm-edge-4b-chat</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>glm-edge-4b-chat</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>gluon_inception_v3</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>gluon_resnet50_v1b</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>gluon_resnext50_32x4d</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>gluon_senet154</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>gluon_seresnext101_64x4d</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>gmcnn-places2</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>gmcnn-places2</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>gmcnn-places2</td>
      <td>tf</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>gmixer_24_224</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>gmlp_s16_224</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>googlenet</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>googlenet-v1</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>googlenet-v1</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>googlenet-v1</td>
      <td>tf</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>googlenet-v2</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>googlenet-v2</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>googlenet-v3</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>googlenet-v3</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>googlenet-v3</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>googlenet-v3</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>googlenet-v3</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>googlenet-v3</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>googlenet-v3</td>
      <td>pytorch</td>
      <td>FP32-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>googlenet-v3</td>
      <td>tf2</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>googlenet-v3</td>
      <td>tf2</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>googlenet-v4</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>googlenet-v4</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>googlenet-v4</td>
      <td>tf</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>gpt2</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>gpt2</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>GPT-2</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>GPT-2</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>GPT-2</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>GPT2Neo1.3BPoints</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>gpt2-wikitext2-int8-onnx-0001</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>gpt-j-6b</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>gpt-j-6b</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>gpt-j-6b</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>gpt-neox-20b</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>gpt-neox-20b</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>gpt-oss-20b</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>gpt-oss-20b</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>granite-3b-code-base</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>granite-3b-code-instruct</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>halo2botnet50ts_256</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>halonet50ts</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>haloregnetz_b</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Text Detection</td>
      <td>handwritten-english-recognition-0001</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Text Detection</td>
      <td>handwritten-english-recognition-0001</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Text Detection</td>
      <td>handwritten-japanese-recognition-0001</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Text Detection</td>
      <td>handwritten-japanese-recognition-0001</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Text Detection</td>
      <td>handwritten-japanese-recognition-0001</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Text Detection</td>
      <td>handwritten-score-recognition-0003</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Text Detection</td>
      <td>handwritten-score-recognition-0003</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Text Detection</td>
      <td>handwritten-score-recognition-0003</td>
      <td>tf</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Text Detection</td>
      <td>handwritten-simplified-chinese-recognition-0001</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Text Detection</td>
      <td>handwritten-simplified-chinese-recognition-0001</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>hardcorenas_d</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>hbonet-0.25</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>hbonet-0.25</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>hbonet-0.5</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>hbonet-0.5</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>hbonet-0.5</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>hbonet-1.0</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>hbonet-1.0</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>hbonet-1.0</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>hg-s8-b1-mpii</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>hg-s8-b1-mpii</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>hg-s8-b1-mpii</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Text to Speech</td>
      <td>hifigan_v1</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>higher-hrnet-w32-512</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>higher-hrnet-w32-512</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>higher-hrnet-w32-512</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Text Detection</td>
      <td>horizontal-text-detection-0001</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Text Detection</td>
      <td>horizontal-text-detection-0001</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Text Detection</td>
      <td>horizontal-text-detection-0001</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>hrnet_w32</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Semantic Segmentation</td>
      <td>hrnet-v2-c1-segmentation</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Semantic Segmentation</td>
      <td>hrnet-v2-c1-segmentation</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Semantic Segmentation</td>
      <td>hrnet-v2-c1-segmentation</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>human-pose-estimation-0002</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>human-pose-estimation-0002</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>human-pose-estimation-0002</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>human-pose-estimation-0003</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>human-pose-estimation-0003</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>human-pose-estimation-0004</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>human-pose-estimation-0004</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>human-pose-estimation-0005</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>human-pose-estimation-0005</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>human-pose-estimation-0006</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>human-pose-estimation-0006</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>human-pose-estimation-0007</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>human-pose-estimation-0007</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>human-pose-estimation-3d-0001</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>human-pose-estimation-3d-0001</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>hybrid-cs-model-mri</td>
      <td>tf2</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>hybrid-cs-model-mri</td>
      <td>tf2</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>hybrid-cs-model-mri</td>
      <td>tf2</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Action Recognition</td>
      <td>i3d-flow</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Action Recognition</td>
      <td>i3d-flow</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Action Recognition</td>
      <td>i3d-flow</td>
      <td>tf</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Action Recognition</td>
      <td>i3d-rgb</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Action Recognition</td>
      <td>i3d-rgb</td>
      <td>tf</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Action Recognition</td>
      <td>i3d-rgb</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>ibert-roberta-base</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>ibert-roberta-base</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>ibert-roberta-base-finetuned-mrpc</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Semantic Segmentation</td>
      <td>icnet-camvid-ava-0001</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Semantic Segmentation</td>
      <td>icnet-camvid-ava-0001</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Semantic Segmentation</td>
      <td>icnet-camvid-ava-0001</td>
      <td>tf</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Semantic Segmentation</td>
      <td>icnet-camvid-ava-sparse-30-0001</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Semantic Segmentation</td>
      <td>icnet-camvid-ava-sparse-30-0001</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Semantic Segmentation</td>
      <td>icnet-camvid-ava-sparse-60-0001</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Semantic Segmentation</td>
      <td>icnet-camvid-ava-sparse-60-0001</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Semantic Segmentation</td>
      <td>icnet-camvid-ava-sparse-60-0001</td>
      <td>tf</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Semantic Segmentation</td>
      <td>icnet-camvid-onnx-0001</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Semantic Segmentation</td>
      <td>icnet-camvid-onnx-0001</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Video Processing </td>
      <td>IFRNet_GoPro</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Video Processing </td>
      <td>IFRNet_Vimeo90K</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>ig_resnext101_32x8d</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>image-retrieval-0001</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>image-retrieval-0001</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>image-retrieval-0001</td>
      <td>tf</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>inception_resnet_v2</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>inception_v3</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>inception_v4</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>inception-resnet-v2</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>inception-resnet-v2</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>inception-resnet-v2</td>
      <td>tf</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>inceptionv3-int8-onnx-0001</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Instance Segmentation</td>
      <td>instance-segmentation-person-0007</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Instance Segmentation</td>
      <td>instance-segmentation-person-0007</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Instance Segmentation</td>
      <td>instance-segmentation-person-0007</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Instance Segmentation</td>
      <td>instance-segmentation-security-0002</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Instance Segmentation</td>
      <td>instance-segmentation-security-0002</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Instance Segmentation</td>
      <td>instance-segmentation-security-0010</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Instance Segmentation</td>
      <td>instance-segmentation-security-0010</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Instance Segmentation</td>
      <td>instance-segmentation-security-0050</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Instance Segmentation</td>
      <td>instance-segmentation-security-0050</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Instance Segmentation</td>
      <td>instance-segmentation-security-0083</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Instance Segmentation</td>
      <td>instance-segmentation-security-0083</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Instance Segmentation</td>
      <td>instance-segmentation-security-0091</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Instance Segmentation</td>
      <td>instance-segmentation-security-0091</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Instance Segmentation</td>
      <td>instance-segmentation-security-0228</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Instance Segmentation</td>
      <td>instance-segmentation-security-0228</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Instance Segmentation</td>
      <td>instance-segmentation-security-1025</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Instance Segmentation</td>
      <td>instance-segmentation-security-1025</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Instance Segmentation</td>
      <td>instance-segmentation-security-1025</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Instance Segmentation</td>
      <td>instance-segmentation-security-1039</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Instance Segmentation</td>
      <td>instance-segmentation-security-1039</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Instance Segmentation</td>
      <td>instance-segmentation-security-1040</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Instance Segmentation</td>
      <td>instance-segmentation-security-1040</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Audio</td>
      <td>intel_dns</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Audio</td>
      <td>intel_dns</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Audio</td>
      <td>intel_dns</td>
      <td>onnx</td>
      <td>INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>internlm2-1_8b</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>internlm2-7b</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>internlm2-chat-1_8b</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>internlm2-chat-7b</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>internlm2-chat-7b-sft</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>internlm2-math-7b</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>internlm2-math-base-7b</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Multimodal Large Language Model</td>
      <td>internvl2-4b</td>
      <td>pytorch</td>
      <td>FP4-NORMALIZED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Multimodal Large Language Model</td>
      <td>internvl2-4b</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Multimodal Large Language Model</td>
      <td>internvl2-4b</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Multimodal Large Language Model</td>
      <td>internvl2-4b</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Video Processing </td>
      <td>iseebetter</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Video Processing </td>
      <td>iseebetter</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Video Processing </td>
      <td>iseebetter</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>jais-13b</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>jais-13b</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>jais-13b</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>jina-clip-v1</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>jx_nest_small</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>kcbert-base</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>keypoint_rcnn_R_50_FPN_1x</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>keypointrcnn_resnet50_fpn</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>klue-roberta-base-ner</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>lambda_resnet50ts</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>lamhalobotnet50ts_256</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>landmarks-regression-retail-0009</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>landmarks-regression-retail-0009</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>landmarks-regression-retail-0009</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>layoutlm-base-uncased</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Generation</td>
      <td>lcm-dreamshaper-v7</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Generation</td>
      <td>lcm-dreamshaper-v7</td>
      <td>pytorch</td>
      <td>INT8-HYBRID</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Generation</td>
      <td>lcm-dreamshaper-v7</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Generation</td>
      <td>lcm-sdxl</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Generation</td>
      <td>lcm-sdxl</td>
      <td>pytorch</td>
      <td>INT8-HYBRID</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Generation</td>
      <td>lcm-sdxl</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>lcnet_075.ra2_in1k</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image to Image</td>
      <td>LearningToPaint</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>learning-to-see-in-the-dark-fuji</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>learning-to-see-in-the-dark-fuji</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>learning-to-see-in-the-dark-fuji</td>
      <td>tf</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>learning-to-see-in-the-dark-sony</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>learning-to-see-in-the-dark-sony</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>learning-to-see-in-the-dark-sony</td>
      <td>tf</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>led-base-16384</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>legacy_senet154</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>legacy_seresnet34</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>legacy_seresnext50_32x4d</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>legal-bert-base-uncased</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Particle Interaction</td>
      <td>lennard_jones</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>levit_128s.fb_dist_in1k</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>levit_conv_128.fb_dist_in1k</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>levit-128s</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>levit-128s</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>levit-128s</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>levit-128s</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>levit-256</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Text Detection</td>
      <td>license-plate-recognition-barrier-0007</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Text Detection</td>
      <td>license-plate-recognition-barrier-0007</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Text Detection</td>
      <td>license-plate-recognition-barrier-0007</td>
      <td>tf</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>llama-2-13b-chat-hf</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>llama-2-13b-chat-hf</td>
      <td>pytorch</td>
      <td>FP4-NORMALIZED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>llama-2-13b-chat-hf</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>llama-2-13b-chat-hf</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>Llama-2-13b-chat-hf</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>Llama-2-13b-hf</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>llama-2-7b-chat-hf</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>llama-2-7b-chat-hf</td>
      <td>pytorch</td>
      <td>FP4-NORMALIZED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>llama-2-7b-chat-hf</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>llama-2-7b-chat-hf</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>Llama-2-7b-chat-hf</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>Llama-2-7b-hf</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>llama-3.1-8b</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>llama-3.1-8b</td>
      <td>pytorch</td>
      <td>FP4-NORMALIZED</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>llama-3.1-8b-instruct</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>llama-3.1-8b-instruct</td>
      <td>pytorch</td>
      <td>FP4-NORMALIZED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>llama-3.1-8b-instruct</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>llama-3.1-8b-instruct</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>llama-3.2-1b-instruct</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>llama-3.2-1b-instruct</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>llama-3.2-1b-instruct</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>llama-3.2-3b-instruct</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>llama-3.2-3b-instruct</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>llama-3.2-3b-instruct</td>
      <td>pytorch</td>
      <td>FP4-NORMALIZED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>llama-3.2-3b-instruct</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>llama-3-8b-instruct</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>llama-3-8b-instruct</td>
      <td>pytorch</td>
      <td>FP4-NORMALIZED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>llama-3-8b-instruct</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>llama-3-8b-instruct</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Vision Language Model</td>
      <td>llava-next-video-7b-hf</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Vision Language Model</td>
      <td>llava-next-video-7b-hf</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Vision Language Model</td>
      <td>llava-next-video-7b-hf</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Vision Language Model</td>
      <td>llava-v1.6-mistral-7b-hf</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Vision Language Model</td>
      <td>llava-v1.6-mistral-7b-hf</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>Longformer</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>longformer-base-4096</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>lraspp_mobilenet_v3_large</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>lraspp-mobilenet-v3-large</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>lraspp-mobilenet-v3-large</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>lraspp-mobilenet-v3-large</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Video Processing </td>
      <td>ltx-video</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Video Processing </td>
      <td>ltx-video</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Video Processing </td>
      <td>ltx-video</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>m2m100_418M</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>machine-translation-nar-de-en-0001</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>machine-translation-nar-de-en-0001</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>machine-translation-nar-de-en-0002</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>machine-translation-nar-de-en-0002</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>machine-translation-nar-en-de-0001</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>machine-translation-nar-en-de-0001</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>machine-translation-nar-en-de-0002</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>machine-translation-nar-en-de-0002</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>machine-translation-nar-en-ru-0002</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>machine-translation-nar-en-ru-0002</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>machine-translation-nar-ru-en-0002</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>machine-translation-nar-ru-en-0002</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>machine-translation-nar-ru-en-0002</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>Magicoder-CL-7B</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>Magicoder-DS-6.7B</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>Magicoder-S-CL-7B</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>Magicoder-S-DS-6.7B</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Text Recognition</td>
      <td>maml_omniglot</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>MarianCausalLM</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>marian-finetuned-kde4-en-to-cn-accelerate</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Instance Segmentation</td>
      <td>mask_rcnn_inception_resnet_v2_atrous_coco</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Instance Segmentation</td>
      <td>mask_rcnn_inception_resnet_v2_atrous_coco</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Instance Segmentation</td>
      <td>mask_rcnn_inception_v2_coco</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Instance Segmentation</td>
      <td>mask_rcnn_inception_v2_coco</td>
      <td>tf</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>mask_rcnn_R_50_C4_1x</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>mask_rcnn_R_50_DC5_1x</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>mask_rcnn_R_50_FPN</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>mask_rcnn_R_50_FPN_1x</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>mask_rcnn_R_50_FPN_3x</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>mask_rcnn_R_50_FPN_3x_dconv_c3-c5</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>mask_rcnn_R_50_FPN_noaug_1x</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Instance Segmentation</td>
      <td>mask_rcnn_resnet101_atrous_coco</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Instance Segmentation</td>
      <td>mask_rcnn_resnet101_atrous_coco</td>
      <td>tf</td>
      <td>INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Instance Segmentation</td>
      <td>mask_rcnn_resnet101_atrous_coco</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Instance Segmentation</td>
      <td>mask_rcnn_resnet101_atrous_coco</td>
      <td>tf</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Instance Segmentation</td>
      <td>mask_rcnn_resnet50_atrous_coco</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Instance Segmentation</td>
      <td>mask_rcnn_resnet50_atrous_coco</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Instance Segmentation</td>
      <td>mask_rcnn_resnet50_atrous_coco</td>
      <td>tf</td>
      <td>INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Instance Segmentation</td>
      <td>mask_rcnn_resnet50_atrous_coco</td>
      <td>tf</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>Mask-AI-SRGAN</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>Mask-AI-SRGAN</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>Mask-AI-SRGAN</td>
      <td>tf</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>maskrcnn_resnet50_fpn</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>maskrcnn_resnet50_fpn_v2</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>mask-rcnn-resnet50-fpn</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>mask-rcnn-resnet50-fpn</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>maxvit_rmlp_small_rw_224.sw_in1k</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>maxvit_t</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>maxxvit_rmlp_small_rw_256.sw_in1k</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>maxxvitv2_rmlp_base_rw_224.sw_in12k_ft_in1k</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>mbart-large-50-many-to-one-mmt</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Action Recognition</td>
      <td>mc3_18</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>Meta-Llama-3-8B</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>Meta-Llama-3-8B-Instruct</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>Meta-Llama-Guard-2-8B</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Monodepth</td>
      <td>midasnet</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Monodepth</td>
      <td>midasnet</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Monodepth</td>
      <td>midasnet</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>MindChat-Qwen2-4B</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>minicpm-1b-sft</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>minicpm-1b-sft</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>minicpm-1b-sft</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>MiniCPM-2B-sft-bf16</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>minicpm3-4b</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>minicpm3-4b</td>
      <td>pytorch</td>
      <td>FP4-NORMALIZED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>minicpm3-4b</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>minicpm3-4b</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>minicpm4-0.5b</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>minicpm4-0.5b</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>minicpm4-0.5b</td>
      <td>pytorch</td>
      <td>FP4-NORMALIZED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>minicpm4-0.5b</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>minicpm4-8b</td>
      <td>pytorch</td>
      <td>FP4-NORMALIZED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>minicpm4-8b</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>minicpm4-8b</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>minicpm4-8b</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Multimodal Large Language Model</td>
      <td>minicpm-o-2_6</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Multimodal Large Language Model</td>
      <td>minicpm-o-2_6</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Multimodal Large Language Model</td>
      <td>minicpm-o-2_6</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Multimodal Large Language Model</td>
      <td>minicpm-v-2_6</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Multimodal Large Language Model</td>
      <td>minicpm-v-2_6</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Multimodal Large Language Model</td>
      <td>minicpm-v-2_6</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Multimodal Large Language Model</td>
      <td>minicpm-v-4_5</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Multimodal Large Language Model</td>
      <td>minicpm-v-4_5</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Multimodal Large Language Model</td>
      <td>minicpm-v-4_5</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>minilm-uncased-squad2</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>mistral-7b-instruct-v0.1</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>mistral-7b-instruct-v0.1</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>mistral-7b-instruct-v0.1</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>mistral-7b-instruct-v0.2</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>mistral-7b-instruct-v0.2</td>
      <td>pytorch</td>
      <td>FP4-NORMALIZED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>mistral-7b-instruct-v0.2</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>mistral-7b-instruct-v0.2</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>Mistral-7B-Instruct-v0.2</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>mistral-7b-instruct-v0.3</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>mistral-7b-instruct-v0.3</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>mistral-7b-instruct-v0.3</td>
      <td>pytorch</td>
      <td>FP4-NORMALIZED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>mistral-7b-instruct-v0.3</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>Mistral-7B-Instruct-v0.3</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>mistral-7b-v0.1</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>mistral-7b-v0.1</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>mistral-7b-v0.1</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>Mistral-7B-v0.3</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>mistral-small-24b-instruct-2501</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>mistral-small-24b-instruct-2501</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>mixer_b16_224_miil</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>mixer_l16_224_in21k</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>mixnet_l.ft_in1k</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>mixnet-l</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>mixnet-l</td>
      <td>pytorch</td>
      <td>FP32-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>mixtral-8x7b-v0.1</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>mixtral-8x7b-v0.1</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>mlp-mixer-b16-224-miil</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>mlp-mixer-b16-224-miil</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>mlp-mixer-b16-224-miil</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>mlp-mixer-b16-224-miil</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>mlp-mixer-b16-224-miil</td>
      <td>pytorch</td>
      <td>FP32-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>mnasnet_small.lamb_in1k</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>mnasnet-0.5</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>mnasnet-0.5</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>mnasnet-0.5</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>mnasnet0_5</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>mnasnet0_75</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>mnasnet-1.0</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>mnasnet-1.0</td>
      <td>pytorch</td>
      <td>FP32-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>mnasnet1_3</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>mobilebert</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>mobilebert-uncased</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>mobilenet_v1_1.0_224</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>mobilenet_v2</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>mobilenet_v2_1.4_224</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>mobilenet_v3_large</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>mobilenet_v3_small</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>mobilenet-v1-0.25-128</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>mobilenet-v1-0.25-128</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>mobilenet-v1-0.25-128</td>
      <td>tf</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>mobilenet-v1-1.0-224</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>mobilenet-v1-1.0-224</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>mobilenet-v1-1.0-224</td>
      <td>tf</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>mobilenet-v2</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>mobilenet-v2</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>mobilenet-v2</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>mobilenet-v2</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>mobilenet-v2</td>
      <td>paddle</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>mobilenet-v2</td>
      <td>paddle</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>mobilenet-v2</td>
      <td>pytorch</td>
      <td>FP32-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>mobilenet-v2</td>
      <td>paddle</td>
      <td>INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>mobilenetv2_050.lamb_in1k</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>mobilenet-v2-1.0-224</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>mobilenet-v2-1.0-224</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>mobilenet-v2-1.0-224</td>
      <td>tf</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>mobilenet-v2-1.0-224</td>
      <td>tf2</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>mobilenet-v2-1.0-224</td>
      <td>tf2</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>mobilenet-v2-1.0-224</td>
      <td>tf2</td>
      <td>INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>mobilenetv2-int8-onnx-0001</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>mobilenetv3_large_100.miil_in21k</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>mobilenet-v3-large-1.0-224</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>mobilenet-v3-large-1.0-224</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>mobilenet-v3-large-1.0-224</td>
      <td>tf2</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>mobilenet-v3-large-1.0-224</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>mobilenet-v3-large-1.0-224</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>mobilenet-v3-large-1.0-224</td>
      <td>tf2</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>mobilenet-v3-large-1.0-224</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>mobilenet-v3-large-1.0-224</td>
      <td>tf2</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>mobilenet-v3-large-1.0-224</td>
      <td>paddle</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>mobilenet-v3-large-1.0-224</td>
      <td>paddle</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>mobilenet-v3-large-1.0-224</td>
      <td>paddle</td>
      <td>INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>mobilenet-v3-small-1.0-224</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>mobilenet-v3-small-1.0-224</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>mobilenet-v3-small-1.0-224</td>
      <td>tf2</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>mobilenet-v3-small-1.0-224</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>mobilenet-v3-small-1.0-224</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>mobilenet-v3-small-1.0-224</td>
      <td>tf2</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>mobilenet-v3-small-1.0-224</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>mobilenet-v3-small-1.0-224</td>
      <td>tf2</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>mobilenet-yolo-v4-syg</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>mobilenet-yolo-v4-syg</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>mobilenet-yolo-v4-syg</td>
      <td>tf</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>mobilevit_s</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>mobilevitv2_150</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>mobilevit-xx-small</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Background Matting</td>
      <td>modnet_photographic_portrait_matting</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Background Matting</td>
      <td>modnet_photographic_portrait_matting</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Background Matting</td>
      <td>modnet_photographic_portrait_matting</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Background Matting</td>
      <td>modnet_webcam_portrait_matting</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Background Matting</td>
      <td>modnet_webcam_portrait_matting</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Background Matting</td>
      <td>modnet_webcam_portrait_matting</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Speech Recognition</td>
      <td>mozilla-deepspeech-0.6.1</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Speech Recognition</td>
      <td>mozilla-deepspeech-0.6.1</td>
      <td>tf</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Speech Recognition</td>
      <td>mozilla-deepspeech-0.6.1</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Speech Recognition</td>
      <td>mozilla-deepspeech-0.7.1</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Speech Recognition</td>
      <td>mozilla-deepspeech-0.7.1</td>
      <td>tf</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Speech Recognition</td>
      <td>mozilla-deepspeech-0.7.1</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Speech Recognition</td>
      <td>mozilla-deepspeech-0.8.2</td>
      <td>tf</td>
      <td>INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Speech Recognition</td>
      <td>mozilla-deepspeech-0.8.2</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Speech Recognition</td>
      <td>mozilla-deepspeech-0.8.2</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Speech Recognition</td>
      <td>mozilla-deepspeech-0.8.2</td>
      <td>tf</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>mpnet-base</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>mpt-30b-chat</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>mpt-30b-chat</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>mpt-7b</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>mpt-7b-8k</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>mpt-7b-8k-chat</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>mpt-7b-chat</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>mpt-7b-instruct</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>mpt-7b-storywriter</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>ms-marco-electra-base</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>mt5-small-sum-de-en-v2</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Video Classification</td>
      <td>mvit_v1_b</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Video Classification</td>
      <td>mvit_v2_s</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Video Classification</td>
      <td>mvitv2_small</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>nanodet-m-1.5x-416</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>nanodet-m-1.5x-416</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>nanodet-m-1.5x-416</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Multimodal Large Language Model</td>
      <td>nanollava</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Multimodal Large Language Model</td>
      <td>nanollava</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Multimodal Large Language Model</td>
      <td>nanollava</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>nasnet-a-large-331</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>nasnet-a-large-331</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>nasnet-a-large-331</td>
      <td>tf</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>nasnet-a-mobile-224</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>nasnet-a-mobile-224</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>nasnet-a-mobile-224</td>
      <td>tf</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>netvlad</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>netvlad</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>netvlad</td>
      <td>tf</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>neural-chat-7b-v1-1</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>neural-chat-7b-v3-3</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>neural-chat-7b-v3-3</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>neural-chat-7b-v3-3</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>neural-chat-7b-v3-3</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>nf_regnet_b1</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>nf_resnet50</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>nfnet</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>nfnet_l0</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>nfnet-f0</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>nfnet-f0</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>nfnet-f0</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>nfnet-f6</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>nfnet-f6</td>
      <td>pytorch</td>
      <td>FP32-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Audio Enhancement</td>
      <td>noise-suppression-denseunet-ll-0001</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Audio Enhancement</td>
      <td>noise-suppression-poconetlike-0001</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Audio Enhancement</td>
      <td>noise-suppression-poconetlike-0001</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Audio Enhancement</td>
      <td>noise-suppression-poconetlike-0001</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>notus-7b-v1</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>notus-7b-v1</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>notus-7b-v1</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Recommender</td>
      <td>nvidia_deeprecommender</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Text Detection</td>
      <td>ocr-perpetuuiti</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Text Detection</td>
      <td>ocr-perpetuuiti</td>
      <td>tf</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Text Detection</td>
      <td>ocr-perpetuuiti</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>open-closed-eye-0001</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>open-closed-eye-0001</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>open-closed-eye-0001</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>opennmt-hindi-english</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>openpose-pose</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>openpose-pose</td>
      <td>tf</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>openpose-pose</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>opt-1.3b</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>opt-2.7b</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Panoptic Segmentation</td>
      <td>panoptic_fpn_R_101_dconv_cascade_gn_3x</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Panoptic Segmentation</td>
      <td>panoptic_fpn_R_50_1x</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>Pegasus-7b-slerp</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>pegasus-samsum</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>persimmon-8b-chat</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>person-attributes-recognition-crossroad-0230</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>person-attributes-recognition-crossroad-0230</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>person-attributes-recognition-crossroad-0234</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>person-attributes-recognition-crossroad-0234</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>person-attributes-recognition-crossroad-0234</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>person-attributes-recognition-crossroad-0238</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>person-attributes-recognition-crossroad-0238</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>person-attributes-recognition-crossroad-0238</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>person-detection-0100</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>person-detection-0100</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>person-detection-0101</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>person-detection-0101</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>person-detection-0102</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>person-detection-0102</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>person-detection-0106</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>person-detection-0106</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>person-detection-0200</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>person-detection-0200</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>person-detection-0201</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>person-detection-0201</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>person-detection-0202</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>person-detection-0202</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>person-detection-0203</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>person-detection-0203</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>person-detection-0203</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>person-detection-0301</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>person-detection-0301</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>person-detection-0302</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>person-detection-0302</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>person-detection-0303</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>person-detection-0303</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>person-detection-action-recognition-0006</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>person-detection-action-recognition-0006</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>person-detection-action-recognition-0006</td>
      <td>tf</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>person-detection-asl-0001</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>person-detection-asl-0001</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>person-detection-asl-0001</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>person-reidentification-retail-0248</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>person-reidentification-retail-0248</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>person-reidentification-retail-0248</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>person-reidentification-retail-0277</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>person-reidentification-retail-0277</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>person-reidentification-retail-0286</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>person-reidentification-retail-0286</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>person-reidentification-retail-0287</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>person-reidentification-retail-0287</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>person-reidentification-retail-0287</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>person-reidentification-retail-0288</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>person-reidentification-retail-0288</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>person-vehicle-bike-detection-2000</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>person-vehicle-bike-detection-2000</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>person-vehicle-bike-detection-2001</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>person-vehicle-bike-detection-2001</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>person-vehicle-bike-detection-2002</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>person-vehicle-bike-detection-2002</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>person-vehicle-bike-detection-2003</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>person-vehicle-bike-detection-2003</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>person-vehicle-bike-detection-2003</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>person-vehicle-bike-detection-2004</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>person-vehicle-bike-detection-2004</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>person-vehicle-bike-detection-crossroad-1016</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>person-vehicle-bike-detection-crossroad-1016</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>phi-2</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>phi-2</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>phi-2</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>phi-2</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>phi-3.5-mini-instruct</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>phi-3.5-mini-instruct</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>phi-3.5-mini-instruct</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>phi-3.5-moe-instruct</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Multimodal Large Language Model</td>
      <td>phi-3.5-vision-instruct</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Multimodal Large Language Model</td>
      <td>phi-3.5-vision-instruct</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Multimodal Large Language Model</td>
      <td>phi-3.5-vision-instruct</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>phi-3-medium-4k-instruct</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>phi-3-medium-4k-instruct</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>phi-3-medium-4k-instruct</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>Phi-3-medium-4k-instruct</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>phi-3-mini-128k-instruct</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>phi-3-mini-128k-instruct</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>phi-3-mini-128k-instruct</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>Phi-3-mini-128k-instruct</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>phi-3-mini-4k-instruct</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>phi-3-mini-4k-instruct</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>phi-3-mini-4k-instruct</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>phi-4</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>phi-4</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>phi-4</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>phi-4-mini-instruct</td>
      <td>pytorch</td>
      <td>FP4-NORMALIZED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>phi-4-mini-instruct</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>phi-4-mini-instruct</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>phi-4-mini-instruct</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>phi-4-mini-reasoning</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>phi-4-mini-reasoning</td>
      <td>pytorch</td>
      <td>FP4-NORMALIZED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>phi-4-mini-reasoning</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>phi-4-mini-reasoning</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Multimodal Large Language Model</td>
      <td>phi-4-multimodal-instruct</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Multimodal Large Language Model</td>
      <td>phi-4-multimodal-instruct</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Multimodal Large Language Model</td>
      <td>phi-4-multimodal-instruct</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>phi-4-reasoning</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>phi-4-reasoning</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>phi-4-reasoning</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>phlippe_densenet</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>phlippe_resnet</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>phobert-base</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>pit_s_224</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Vision Language Model</td>
      <td>pix2struct-base</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Segmentation</td>
      <td>pointrend-resnet50-fpn-pytorch</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Segmentation</td>
      <td>pointrend-resnet50-fpn-pytorch</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Segmentation</td>
      <td>pointrend-resnet50-fpn-pytorch</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>Polish_RoBERTa_large_OPI</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>poolformer_s24</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>pp-ocr-det</td>
      <td>paddle</td>
      <td>INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>pp-ocr-det</td>
      <td>paddle</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>pp-ocr-det</td>
      <td>paddle</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>pp-ocr-rec</td>
      <td>paddle</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>pp-ocr-rec</td>
      <td>paddle</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>pp-ocr-rec</td>
      <td>paddle</td>
      <td>INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>pp-yolo</td>
      <td>paddle</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>pp-yolo</td>
      <td>paddle</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>pp-yolo</td>
      <td>paddle</td>
      <td>INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>prnet</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>prnet</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>prnet</td>
      <td>tf</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>product-detection-0001</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>product-detection-0001</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>product-detection-0001</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>prot_bert</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Semantic Segmentation</td>
      <td>pspnet-pytorch</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Semantic Segmentation</td>
      <td>pspnet-pytorch</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Semantic Segmentation</td>
      <td>pspnet-pytorch</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>PubChem10M_SMILES_BPE_396_250</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>pvt_v2_b3</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Ocean Simulator</td>
      <td>pyhpc_equation_of_state</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Ocean Simulator</td>
      <td>pyhpc_turbulent_kinetic_energy</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>pythia-1.4b</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>pythia-12b</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>pythia-12b</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>pythia-12b</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>pythia-12b</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>pythia-14m</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>pythia-160m</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>pythia-1b</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>pythia-2.8b</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>pythia-410m</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>pythia-6.9b</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>pythia-70m</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>Pythia-Chat-Base-7B</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image to Image</td>
      <td>pytorch_CycleGAN_and_pix2pix</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image to Image</td>
      <td>pytorch_stargan</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Speech Recognition</td>
      <td>quartznet-15x5-en</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Speech Recognition</td>
      <td>quartznet-15x5-en</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Speech Recognition</td>
      <td>quartznet-15x5-en</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>Qwen1.5-0.5B</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>Qwen1.5-0.5B-Chat</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>Qwen1.5-1.8B</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>Qwen1.5-1.8B-Chat</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>qwen1.5-14b-chat</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>qwen1.5-14b-chat</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>qwen1.5-14b-chat</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>Qwen1.5-4B</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>Qwen1.5-4B-Chat</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>Qwen1.5-7B</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>Qwen1.5-7B-Chat</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>qwen2.5-1.5b-instruct</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>qwen2.5-1.5b-instruct</td>
      <td>pytorch</td>
      <td>FP4-NORMALIZED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>qwen2.5-1.5b-instruct</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>qwen2.5-1.5b-instruct</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>qwen2.5-3b-instruct-gptq</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>qwen2.5-7b-instruct</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>qwen2.5-7b-instruct</td>
      <td>pytorch</td>
      <td>FP4-NORMALIZED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>qwen2.5-7b-instruct</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>qwen2.5-7b-instruct</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>qwen2.5-7b-instruct-1m</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>qwen2.5-7b-instruct-1m</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>qwen2.5-7b-instruct-1m</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>qwen2.5-coder-0.5b-instruct</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>qwen2.5-coder-0.5b-instruct</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>qwen2.5-coder-0.5b-instruct</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>qwen2.5-coder-1.5b-instruct</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>qwen2.5-coder-1.5b-instruct</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>qwen2.5-coder-1.5b-instruct</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>qwen2.5-coder-3b-instruct</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>qwen2.5-coder-3b-instruct</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>qwen2.5-coder-3b-instruct</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Multimodal Large Language Model</td>
      <td>qwen2.5-vl-3b-instruct</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Multimodal Large Language Model</td>
      <td>qwen2.5-vl-3b-instruct</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Multimodal Large Language Model</td>
      <td>qwen2.5-vl-3b-instruct</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Multimodal Large Language Model</td>
      <td>qwen2.5-vl-7b-instruct</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Multimodal Large Language Model</td>
      <td>qwen2.5-vl-7b-instruct</td>
      <td>pytorch</td>
      <td>FP4-NORMALIZED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Multimodal Large Language Model</td>
      <td>qwen2.5-vl-7b-instruct</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Multimodal Large Language Model</td>
      <td>qwen2.5-vl-7b-instruct</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>qwen2-0.5b</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>qwen2-0.5b</td>
      <td>pytorch</td>
      <td>BF16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>qwen2-0.5b</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>qwen2-0.5b</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>qwen2-1.5b</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>qwen2-1.5b</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>qwen2-1.5b</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>qwen2-7b-instruct</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>qwen2-7b-instruct</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>qwen2-7b-instruct</td>
      <td>pytorch</td>
      <td>FP4-NORMALIZED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>qwen2-7b-instruct</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>qwen2-7b-instruct-gptq</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Multimodal Large Language Model</td>
      <td>qwen2-vl-7b-instruct</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Multimodal Large Language Model</td>
      <td>qwen2-vl-7b-instruct</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Multimodal Large Language Model</td>
      <td>qwen2-vl-7b-instruct</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>qwen3.5-0.8b</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>qwen3.5-0.8b</td>
      <td>pytorch</td>
      <td>FP4-NORMALIZED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>qwen3.5-0.8b</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>qwen3.5-0.8b</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>qwen3.5-27b</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>qwen3.5-35b-a3b</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>qwen3.5-9b</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>qwen3.5-9b</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>qwen3.5-9b</td>
      <td>pytorch</td>
      <td>FP4-NORMALIZED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>qwen3.5-9b</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>qwen3.6-27b</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>qwen3.6-35b-a3b</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>qwen3_8b_eagle3</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>qwen3_8b_eagle3</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>qwen3_8b_eagle3</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>qwen3-30b-a3b</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>qwen3-30b-a3b</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>qwen3-4b</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>qwen3-4b</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>qwen3-4b</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>qwen3-8b</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>qwen3-8b</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>qwen3-8b</td>
      <td>pytorch</td>
      <td>FP4-NORMALIZED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>qwen3-8b</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>qwen3-embedding-0.6b</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>qwen3-embedding-0.6b</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>qwen3-embedding-0.6b</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>qwen3-reranker-0.6b</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>qwen3-reranker-0.6b</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>qwen3-reranker-0.6b</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>qwen3-reranker-0.6b-seq-cls</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>qwen3-reranker-0.6b-seq-cls</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>qwen3-reranker-0.6b-seq-cls</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Multimodal Large Language Model</td>
      <td>qwen3-vl-4b-instruct</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Multimodal Large Language Model</td>
      <td>qwen3-vl-4b-instruct</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Multimodal Large Language Model</td>
      <td>qwen3-vl-4b-instruct</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>qwen-7b-chat</td>
      <td>pytorch</td>
      <td>FP4-NORMALIZED</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>qwen-7b-chat</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>qwen-7b-chat</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>qwen-7b-chat</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Video Classification</td>
      <td>r2plus1d_18</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Video Classification</td>
      <td>r3d_18</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Movement Prediction</td>
      <td>raft_large</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Movement Prediction</td>
      <td>raft_small</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>rcan</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>rcan</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>rcan</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>RedPajama-INCITE-7B-Base</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>RedPajama-INCITE-7B-Chat</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>RedPajama-INCITE-7B-Instruct</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>RedPajama-INCITE-Chat-3B-v1</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>regnet</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>regnet_x_400mf</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>regnet_x_800mf</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>regnet_x_8gf</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>regnet_y_128gf</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>regnet_y_400mf</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>regnet_y_800mf</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>regnet_y_8gf</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>regnetv_040</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>regnetx_032</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>RegNetX_32GF_dds_8gpu</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>regnetx-3.2gf</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>regnetx-3.2gf</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>regnetx-3.2gf</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>regnetx-32gf</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>regnetx-32gf</td>
      <td>pytorch</td>
      <td>FP32-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>regnety_080</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>regnetz_b16</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>repvgg_b3</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>repvgg_b3.rvgg_in1k</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>repvgg-a0</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>repvgg-a0</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>repvgg-b1</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>repvgg-b1</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>repvgg-b3</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>repvgg-b3</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>repvgg-b3</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>repvgg-b3</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>repvgg-b3</td>
      <td>pytorch</td>
      <td>FP32-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>res2net50_26w_4s</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>res2next50</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resmlp_12_224.fb_distilled_in1k</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resmlp_12_distilled_224</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resmlp-12-distilled-224</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resmlp-12-distilled-224</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resmlp-12-distilled-224</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resmlp-12-distilled-224</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resmlp-12-distilled-224</td>
      <td>pytorch</td>
      <td>FP32-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resnest</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resnest200e</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resnest50d_4s2x40d</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resnest-50-pytorch</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resnest-50-pytorch</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resnest-50-pytorch</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resnet101</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resnet-101</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resnet-101</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resnet-101</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resnet-101</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resnet152</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resnet-152</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resnet-152</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resnet-152</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resnet-152</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resnet18</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resnet-18</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resnet-18</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resnet-18</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resnet-18</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resnet18-xnor-binary-onnx-0001</td>
      <td>onnx</td>
      <td>FP16-INT1</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resnet34</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resnet-34-pytorch</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resnet-34-pytorch</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resnet50</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resnet-50</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resnet-50</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resnet-50</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resnet-50</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resnet-50</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resnet-50</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resnet-50</td>
      <td>tf</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resnet-50</td>
      <td>paddle</td>
      <td>INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resnet-50</td>
      <td>paddle</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resnet-50</td>
      <td>paddle</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resnet-50</td>
      <td>pytorch</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resnet50-binary-0001</td>
      <td>onnx</td>
      <td>FP16-INT1</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resnet50d</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resnet50-int8-onnx-0001</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resnet-50-pytorch</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resnet-50-pytorch</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resnet-50-pytorch</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resnet-50-tf</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resnet-50-tf</td>
      <td>tf2</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resnet-50-tf</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resnet-50-tf</td>
      <td>tf</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resnet-50-tf</td>
      <td>tf2</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.2</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resnet-50-tf</td>
      <td>tf2</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resnetaa50</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resnetrs200</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resnet-v1.5-50</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resnet-v1.5-50</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resnet-v1.5-50</td>
      <td>tf</td>
      <td>INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resnetv2_101x1_bitm</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resnext_101_32x8d</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resnext101_64x4d</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resnext101-32x16d-swsl</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resnext101-32x16d-swsl</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resnext101-32x16d-swsl</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resnext-101-32x8d</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resnext-101-32x8d</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resnext-101-32x8d</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resnext-101-32x8d</td>
      <td>pytorch</td>
      <td>FP32-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resnext50_32x4d</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resnext-50-32x4d</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resnext-50-32x4d</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>resnext-50-32x4d</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>retinaface-resnet50</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>retinaface-resnet50</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>retinaface-resnet50</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>retinanet</td>
      <td>tf2</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>retinanet</td>
      <td>tf2</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>retinanet</td>
      <td>tf</td>
      <td>INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>retinanet</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>retinanet</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>retinanet_resnet34</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>retinanet_resnet34</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>retinanet_resnet34</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>retinanet_resnet50_fpn</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>retinanet_resnet50_fpn_v2</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>rexnet_130</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>rexnet-v1-x1.0</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>rexnet-v1-x1.0</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>rexnet-v1-x1.0</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>rfcn-resnet101-coco</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>rfcn-resnet101-coco</td>
      <td>tf</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Speech Recognition</td>
      <td>rnnt</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Speech Recognition</td>
      <td>rnnt</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Audio</td>
      <td>rnnt_encoder</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Audio</td>
      <td>rnnt_encoder</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Audio</td>
      <td>rnnt_joint</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Audio</td>
      <td>rnnt_joint</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Audio</td>
      <td>rnnt_prediction</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Audio</td>
      <td>rnnt_prediction</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>roberta-base</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>roberta-base</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>roberta-base</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>roberta-base</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>roberta-base</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>roberta-base-mrpc</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>roberta-base-mrpc</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>roberta-base-mrpc</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>roberta-base-MRPC</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>roberta-base-squad2</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>roberta-large</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>roberta-large</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>roberta-large</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>roberta-large</td>
      <td>pytorch</td>
      <td>FP32-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>roberta-large</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>roberta-large</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>roberta-large-mnli</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>roberta-large-ner-english</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>roberta-large-squadv2</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>robust-video-matting-mobilenetv3</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>robust-video-matting-mobilenetv3</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>roformer_chinese_base</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>roformer_small_generator</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Video Classification</td>
      <td>s3d</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>sbert-base-mean-tokens</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>sbert-base-mean-tokens</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>sbert-base-mean-tokens</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Generation</td>
      <td>sd-turbo</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Generation</td>
      <td>sd-turbo</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Generation</td>
      <td>sd-turbo</td>
      <td>pytorch</td>
      <td>INT8-HYBRID</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>sebotnet33ts_256</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>sehalonet33ts</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>selecsls42b</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>semnasnet_075.rmsp_in1k</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>sequencer2d_m</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>seresnet50</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>se-resnext-101</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>se-resnext-101</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>se-resnext-101</td>
      <td>tf</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>se-resnext-50</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>se-resnext-50</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>se-resnext-50</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>se-resnext-50</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>se-resnext-50</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>seresnext50_32x4d</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>Sharpen-LensBlur</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>Sharpen-LensBlur</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>Sharpen-LensBlur</td>
      <td>tf</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>Sharpen-MotionBlur</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>Sharpen-MotionBlur</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>Sharpen-MotionBlur</td>
      <td>tf</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>Sharpen-Sharpen</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>Sharpen-Sharpen</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>Sharpen-Sharpen</td>
      <td>tf</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>shufflenet</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>shufflenet</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>shufflenet</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>shufflenet_v2_x0_5</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>shufflenet_v2_x1_5</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>shufflenet_v2_x2_0</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>shufflenet-v2-x0.5</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>shufflenet-v2-x0.5</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>shufflenet-v2-x1.0</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>shufflenet-v2-x1.0</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>shufflenet-v2-x1.0</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>shufflenet-v2-x1.0</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>shufflenet-v2-x1.0</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>shufflenet-v2-x1.0</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>shufflenet-v2-x1.0</td>
      <td>tf</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>shufflenet-v2-x1.0</td>
      <td>pytorch</td>
      <td>FP32-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>simmim_swin_transformer</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>single-human-pose-estimation-0001</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>single-human-pose-estimation-0001</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>single-image-super-resolution-1032</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>single-image-super-resolution-1032</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>single-image-super-resolution-1033</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>single-image-super-resolution-1033</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>skresnet34</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>skresnext50_32x4d</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>small-base</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>smartlab-object-detection-0001</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>smartlab-object-detection-0001</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>smartlab-object-detection-0002</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>smartlab-object-detection-0003</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>smartlab-object-detection-0003</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>smartlab-object-detection-0004</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>smartlab-object-detection-0004</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>smartlab-object-detection-0004</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Speech Recognition</td>
      <td>speech_transformer</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>splinter-base</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>spnasnet_100.rmsp_in1k</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>spnasnet-100</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>spnasnet-100</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>spnasnet-100</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>spnasnet-100</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>spnasnet-100</td>
      <td>pytorch</td>
      <td>FP32-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>squeezebert-uncased</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>squeezenet1.0</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>squeezenet1.0</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>squeezenet1.0</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>squeezenet1.0</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>squeezenet1.0</td>
      <td>pytorch</td>
      <td>FP32-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>squeezenet1.1</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>squeezenet1.1</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>squeezenet1.1</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>squeezenet1.1</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>squeezenet1.1</td>
      <td>tf</td>
      <td>INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>squeezenet1.1-pytorch1.2.0</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>squeezenet1.1-pytorch1.2.0</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>squeezenet1.1-pytorch1.2.0</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>squeezenet1_1</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>srgan</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>srgan</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>srgan</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>srgan</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>srgan</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>srgan</td>
      <td>tf</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>ssd_mobilenet_v1_coco</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>ssd_mobilenet_v1_coco</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>ssd_mobilenet_v1_coco</td>
      <td>tf</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>ssd_mobilenet_v1_fpn_coco</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>ssd_mobilenet_v2_coco</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>ssd_mobilenet_v2_coco</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>ssd_mobilenet_v2_coco</td>
      <td>tf</td>
      <td>INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>ssd_mobilenet_v2_coco</td>
      <td>tf</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>ssd_resnet50_v1_fpn_coco</td>
      <td>tf2</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>ssd_resnet50_v1_fpn_coco</td>
      <td>tf2</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>ssd300_vgg16</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>ssd300-int8-onnx-0001</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>ssd300-onnx-0001</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>ssd300-onnx-0001</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>ssd300-onnx-0001</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>ssdlite_mobilenet_v2</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>ssdlite_mobilenet_v2</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>ssdlite_mobilenet_v2</td>
      <td>tf</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>ssdlite320_mobilenet_v3_large</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>ssdlite-mobilenet-v3-small-320-coco</td>
      <td>paddle</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>ssdlite-mobilenet-v3-small-320-coco</td>
      <td>paddle</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>ssd-resnet34-1200</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>ssd-resnet34-1200</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>ssd-resnet34-1200</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>ssd-resnet34-300x300</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>ssd-resnet34-300x300</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>ssd-resnet34-300x300</td>
      <td>tf</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>ssd-vgg16</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>ssd-vgg16</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>ssd-vgg16</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>ssl_resnext101_32x8d</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Generation</td>
      <td>stable-diffusion-3.5-large-turbo</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Generation</td>
      <td>stable-diffusion-3.5-large-turbo</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Generation</td>
      <td>stable-diffusion-3.5-large-turbo</td>
      <td>pytorch</td>
      <td>INT8-HYBRID</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Generation</td>
      <td>stable-diffusion-3.5-large-turbo</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Generation</td>
      <td>stable-diffusion-3.5-medium</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Generation</td>
      <td>stable-diffusion-3.5-medium</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Generation</td>
      <td>stable-diffusion-3.5-medium</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Generation</td>
      <td>stable-diffusion-3.5-medium</td>
      <td>pytorch</td>
      <td>INT8-HYBRID</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Generation</td>
      <td>stable-diffusion-v1-4-text-encoder</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Generation</td>
      <td>stable-diffusion-v1-4-text-encoder</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Generation</td>
      <td>stable-diffusion-v1-4-unet</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Generation</td>
      <td>stable-diffusion-v1-4-unet</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Generation</td>
      <td>stable-diffusion-v1-4-vae-decoder</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Generation</td>
      <td>stable-diffusion-v1-4-vae-decoder</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Generation</td>
      <td>stable-diffusion-v1-4-vae-encoder</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Generation</td>
      <td>stable-diffusion-v1-4-vae-encoder</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Generation</td>
      <td>stable-diffusion-v1-5</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Generation</td>
      <td>stable-diffusion-v1-5</td>
      <td>pytorch</td>
      <td>FP8-STATIC</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Generation</td>
      <td>stable-diffusion-v1-5</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Generation</td>
      <td>stable-diffusion-v1-5</td>
      <td>pytorch</td>
      <td>INT8-HYBRID</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Generation</td>
      <td>stable-diffusion-v2-1</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Generation</td>
      <td>stable-diffusion-v2-1</td>
      <td>pytorch</td>
      <td>INT8-HYBRID</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Generation</td>
      <td>stable-diffusion-v2-1</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Generation</td>
      <td>stable-diffusion-xl-1.0-inpainting-0.1</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Generation</td>
      <td>stable-diffusion-xl-1.0-inpainting-0.1</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>stablelm-2-1_6b</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>stablelm-2-1_6b-chat</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>stablelm-2-12b</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>stablelm-2-12b-chat</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>stablelm-2-zephyr-1_6b</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>stablelm-3b-4e1t</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>stablelm-3b-4e1t</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>stablelm-3b-4e1t</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>stablelm-3b-4e1t</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>stablelm-base-alpha-3b</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>stablelm-tuned-alpha-7b</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>stablelm-zephyr-3b</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>stable-zephyr-3b-dpo</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>stable-zephyr-3b-dpo</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>stable-zephyr-3b-dpo</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>stacked_hourglass</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>starcoder</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>starcoder</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>starcoder</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>starcoder2-15b</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>starcoder2-15b</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>starcoder2-15b</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Spatial Transformer Network</td>
      <td>STN</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Spatial Transformer Network</td>
      <td>STN</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Spatial Transformer Network</td>
      <td>STN</td>
      <td>tf</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Processing </td>
      <td>style_transfer</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Processing </td>
      <td>style_transfer</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Processing </td>
      <td>style_transfer</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Processing </td>
      <td>stylegan2</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Processing </td>
      <td>stylegan2</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Processing </td>
      <td>stylegan2</td>
      <td>tf</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Video Processing </td>
      <td>Super_SloMo</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>sup-simcse-roberta-base</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>swin_b</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>swin_base_patch4_window7_224</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>swin_s</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>swin_t</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>swin_tiny_patch4_window7_224</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>swin_tiny_patch4_window7_224.ms_in1k</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>swin_v2_b</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>swin_v2_s</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>swin_v2_t</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>swin2sr-realworld-sr-x4-64-bsrgan-psnr</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>swin3d_b</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>swin3d_s</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>swin3d_t</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>swin-tiny-patch4-window7-224</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>swin-tiny-patch4-window7-224</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>swin-tiny-patch4-window7-224</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>swin-tiny-patch4-window7-224</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>swinv2_small_window8_256</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>swsl_resnet50</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>swsl_resnext101_32x4d</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>t2t-vit-14</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>t2t-vit-14</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>t2t-vit-14</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>t2t-vit-7</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>t2t-vit-7</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>t2t-vit-7</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>t5-base</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>t5-small</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>t5-small</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>t5-small</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>t5-small</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Text to Speech</td>
      <td>tacotron_2</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Text to Speech</td>
      <td>tacotron_2</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Text Detection</td>
      <td>text-detection-0003</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Text Detection</td>
      <td>text-detection-0003</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Text Detection</td>
      <td>text-detection-0004</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Text Detection</td>
      <td>text-detection-0004</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Text Detection</td>
      <td>text-detection-0004</td>
      <td>tf</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>text-image-super-resolution-0001</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>text-image-super-resolution-0001</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Text Detection</td>
      <td>text-recognition-0012</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Text Detection</td>
      <td>text-recognition-0012</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Text Detection</td>
      <td>text-recognition-0012</td>
      <td>tf</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Text Detection</td>
      <td>text-recognition-0013</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Text Detection</td>
      <td>text-recognition-0013</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Text Detection</td>
      <td>text-recognition-0014</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Text Detection</td>
      <td>text-recognition-0014</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Text Detection</td>
      <td>text-recognition-0015</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Text Detection</td>
      <td>text-recognition-0015</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Text Detection</td>
      <td>text-recognition-0016</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Text Detection</td>
      <td>text-recognition-0016</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Text Detection</td>
      <td>text-recognition-resnet-fc</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Text Detection</td>
      <td>text-recognition-resnet-fc</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Text Detection</td>
      <td>text-spotting-0001</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Text Detection</td>
      <td>text-spotting-0001</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Text Detection</td>
      <td>text-spotting-0003</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Text Detection</td>
      <td>text-spotting-0003</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Text Detection</td>
      <td>text-spotting-0004</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Text Detection</td>
      <td>text-spotting-0004</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Text Detection</td>
      <td>text-spotting-0005</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Text Detection</td>
      <td>text-spotting-0005</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Text to Speech</td>
      <td>text-to-speech-en-0001</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Text to Speech</td>
      <td>text-to-speech-en-0001</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Text to Speech</td>
      <td>text-to-speech-en-multi-0001</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Text to Speech</td>
      <td>text-to-speech-en-multi-0001</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>tf_inception_v3</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Time Series Forecasting</td>
      <td>time-series-forecasting-electricity-0001</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Time Series Forecasting</td>
      <td>time-series-forecasting-electricity-0001</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Time Series Forecasting</td>
      <td>time-series-forecasting-electricity-0001</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>tiny_sd_text_encoder</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Generation</td>
      <td>tiny_sd_unet</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Generation</td>
      <td>tiny_sd_vae_encoder</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>tinybert_6layer_768dim_cola</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>tinybert_6layer_768dim_cola</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>tinybert_6layer_768dim_cola</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>tiny-llama-1.1b-chat</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>tiny-llama-1.1b-chat</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>tiny-llama-1.1b-chat</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>tinynet_d.in1k</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>tiramisu-fc-densenet-103</td>
      <td>tf2</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>tiramisu-fc-densenet-103</td>
      <td>tf2</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>tiramisu-fc-densenet-103</td>
      <td>tf2</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>topaz_video_super_resolution</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>topaz_video_super_resolution</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>tresnet-m</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>tresnet-m</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>tresnet-m</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>trinity-mini</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>trinity-mini</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image to Text</td>
      <td>trocr-base-handwritten</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Text to Speech</td>
      <td>tts_angular</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>tv_densenet121</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>tv_resnet50</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>twins_pcpvt_base</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>twins_pcpvt_large</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>twins_pcpvt_small</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>twins_svt_base</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>twins_svt_large</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>twins_svt_small</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>twitter-roberta-base-sentiment-latest</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>ultra-lightweight-face-detection-rfb-320</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>ultra-lightweight-face-detection-rfb-320</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>ultra-lightweight-face-detection-slim-320</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>ultra-lightweight-face-detection-slim-320</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>ultra-lightweight-face-detection-slim-320</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Semantic Segmentation</td>
      <td>unet-2d</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Semantic Segmentation</td>
      <td>unet-2d</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Semantic Segmentation</td>
      <td>unet-2d</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Semantic Segmentation</td>
      <td>unet3d_mlperf</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Semantic Segmentation</td>
      <td>unet3d_mlperf</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Semantic Segmentation</td>
      <td>unet3d_mlperf</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Semantic Segmentation</td>
      <td>unet3d_mlperf</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Semantic Segmentation</td>
      <td>unet3d_mlperf</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Semantic Segmentation</td>
      <td>unet3d_mlperf</td>
      <td>tf</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Semantic Segmentation</td>
      <td>unet-camvid-int8-onnx-0001</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Semantic Segmentation</td>
      <td>unet-camvid-onnx-0001</td>
      <td>onnx</td>
      <td>INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Semantic Segmentation</td>
      <td>unet-camvid-onnx-0001</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Semantic Segmentation</td>
      <td>unet-camvid-onnx-0001</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Semantic Segmentation</td>
      <td>unet-camvid-onnx-0001</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Audio Classification</td>
      <td>unispeech-sat-base-plus-sd</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Audio</td>
      <td>unispeech-sat-base-sv</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Audio</td>
      <td>unispeech-sat-large-sv</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>URL Classification</td>
      <td>urlnet</td>
      <td>tf2</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>URL Classification</td>
      <td>urlnet</td>
      <td>tf2</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>vehicle-attributes-recognition-barrier-0042</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>vehicle-attributes-recognition-barrier-0042</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>vehicle-attributes-recognition-barrier-0042</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>vehicle-detection-0200</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>vehicle-detection-0200</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>vehicle-detection-0201</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>vehicle-detection-0201</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>vehicle-detection-0202</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>vehicle-detection-0202</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>vehicle-reid-0001</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>vehicle-reid-0001</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>vehicle-reid-0001</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>versa</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>versa</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>versa</td>
      <td>tf</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>vfnet-resnet50-fpn-pytorch</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>vfnet-resnet50-fpn-pytorch</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>vfnet-resnet50-fpn-pytorch</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>vgg11</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>vgg-11-bn</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>vgg-11-bn</td>
      <td>pytorch</td>
      <td>FP32-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>vgg13</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>vgg13_bn</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>vgg16</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>vgg16</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>vgg16</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>vgg16</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>vgg16</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>vgg16</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>vgg16</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>vgg16</td>
      <td>pytorch</td>
      <td>FP32-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>vgg16_bn</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>vgg-16-bn</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>vgg-16-bn</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>vgg-16-bn</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>vgg19</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>vgg19</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>vgg19</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>vgg19</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>vgg19</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>vgg19_bn</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Speech Recognition</td>
      <td>vggvox</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Speech Recognition</td>
      <td>vggvox</td>
      <td>tf</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Speech Recognition</td>
      <td>vggvox</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>visformer_small</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>vision_maskrcnn</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>vit_b_16</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>vit_b_32</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>vit_base_patch16_224.orig_in21k_ft_in1k</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>vit_h_14</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>vit_l_16</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>vit_l_32</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>vit-base-16-224</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>vit-base-16-224</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>vit-base-16-224</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>vit-base-16-224</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>vit-base-16-224</td>
      <td>pytorch</td>
      <td>FP32-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>vit-base-patch16-224</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>vit-base-patch16-224</td>
      <td>pytorch</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>vit-base-patch16-224</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Text Detection</td>
      <td>vitstr-small-patch16-224</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Text Detection</td>
      <td>vitstr-small-patch16-224</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Text Detection</td>
      <td>vitstr-small-patch16-224</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>vovnet</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Speech Recognition</td>
      <td>wav2vec2-base</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Speech Recognition</td>
      <td>wav2vec2-base</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Speech Recognition</td>
      <td>wav2vec2-base</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Audio Classification</td>
      <td>wav2vec2-base-superb-sd</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Audio Classification</td>
      <td>wav2vec2-base-superb-sid</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Audio</td>
      <td>wav2vec2-base-superb-sv</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Text to Speech</td>
      <td>wavernn-rnn</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Text to Speech</td>
      <td>wavernn-rnn</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Text to Speech</td>
      <td>wavernn-rnn</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Text to Speech</td>
      <td>wavernn-upsampler</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Text to Speech</td>
      <td>wavernn-upsampler</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Text to Speech</td>
      <td>wavernn-upsampler</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>wdsr-small-x4</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>wdsr-small-x4</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Processing, Enhancement </td>
      <td>wdsr-small-x4</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>weld-porosity-detection-0001</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>weld-porosity-detection-0001</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>weld-porosity-detection-0001</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>whisper_decoder</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>whisper_encoder</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Speech Recognition</td>
      <td>whisper-base</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Speech Recognition</td>
      <td>whisper-large-v2</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Speech Recognition</td>
      <td>whisper-large-v3</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Speech Recognition</td>
      <td>whisper-large-v3</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Speech Recognition</td>
      <td>whisper-large-v3</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Speech Recognition</td>
      <td>whisper-large-v3-genai</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Speech Recognition</td>
      <td>whisper-large-v3-genai</td>
      <td>pytorch</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Speech Recognition</td>
      <td>whisper-large-v3-turbo</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Speech Recognition</td>
      <td>whisper-large-v3-turbo</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Speech Recognition</td>
      <td>whisper-large-v3-turbo</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Speech Recognition</td>
      <td>whisper-large-v3-turbo-genai</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Speech Recognition</td>
      <td>whisper-large-v3-turbo-genai</td>
      <td>pytorch</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Speech Recognition</td>
      <td>whisper-medium</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Speech Recognition</td>
      <td>whisper-small</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Speech Recognition</td>
      <td>whisper-small</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Speech Recognition</td>
      <td>whisper-small</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Speech Recognition</td>
      <td>whisper-small</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Speech Recognition</td>
      <td>whisper-small-genai</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Speech Recognition</td>
      <td>whisper-small-genai</td>
      <td>pytorch</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Speech Recognition</td>
      <td>whisper-tiny</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Speech Recognition</td>
      <td>whisper-tiny</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Speech Recognition</td>
      <td>whisper-tiny</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Speech Recognition</td>
      <td>whisper-tiny-genai</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Speech Recognition</td>
      <td>whisper-tiny-genai</td>
      <td>pytorch</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>wide_resnet101_2</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>wide_resnet50_2</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>wide-resnet-50-v2</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>wide-resnet-50-v2</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>wide-resnet-50-v2</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>WizardMath-7B-V1.1</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>xception</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>xception65</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>xcit-nano-12-p16</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>xcit-nano-12-p16</td>
      <td>pytorch</td>
      <td>FP32-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>xlm-roberta-base</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>xlm-roberta-base</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>xlm-roberta-base</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>xlm-roberta-base</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>xlm-roberta-base-language-detection</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>xlm-roberta-base-uncased-conll2003</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>xlm-roberta-large</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>xlm-roberta-large-en-ru-mnli</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>xlm-roberta-large-finetuned-conll03-english</td>
      <td>pytorch</td>
      <td>intel-optimum default</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>xlnet</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>xlnet</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>xlnet</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>xlnet-base-cased</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Instance Segmentation</td>
      <td>yolact-resnet50-fpn-pytorch</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Instance Segmentation</td>
      <td>yolact-resnet50-fpn-pytorch</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Instance Segmentation</td>
      <td>yolact-resnet50-fpn-pytorch</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>yolo_v3_tiny</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>yolo_v3_tiny</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>yolo_v3_tiny</td>
      <td>tf</td>
      <td>INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>yolo_v3_tiny</td>
      <td>onnx</td>
      <td>INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>yolo_v3_tiny</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>yolo_v3_tiny</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>yolo_v4</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>yolo_v4</td>
      <td>tf2</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>yolo_v4</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>yolo_v4</td>
      <td>tf2</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>yolo_v4</td>
      <td>tf</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>yolo_v4</td>
      <td>tf2</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>yolo_v5l</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>yolo_v5l</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>yolo_v5l</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>yolo_v5m</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>yolo_v5m</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>yolo_v5m</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>yolo_v5m6-v6</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>yolo_v5m6-v6</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>yolo_v5m-v6</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>yolo_v5m-v6</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>yolo_v5m-v6</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>yolo_v5s</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>yolo_v5s</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>yolo_v5s</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>yolo_v5s-v6</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>yolo_v5s-v6</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>yolo_v5s-v6</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>yolo_v5x</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>yolo_v5x</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>yolo_v8n</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>yolo_v8n</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>yolo_v8n</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>yolo11</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>yolo11</td>
      <td>pytorch</td>
      <td>FP32-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>yolof</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>yolof</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>yolof</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>yolor_p6</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>yolor_p6</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>yolor_p6</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>yolo-v3</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>yolo-v3</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>yolo-v4-tiny</td>
      <td>tf</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>yolo-v4-tiny</td>
      <td>tf</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>yolov5l</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>yolov5l6</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>yolov5m</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>yolov5m6</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>yolov5n</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>yolov5n6</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>yolov5s</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>yolov5s6</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>yolov5x</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>yolov5x6</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>torch.compile</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>yolox-tiny</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>yolox-tiny</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>yolox-tiny</td>
      <td>onnx</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>zephyr-7b-beta</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>zephyr-7b-beta</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>zephyr-7b-beta</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>zephyr-7b-beta</td>
      <td>pytorch</td>
      <td>INT4</td>
      <td>Optimum Intel</td>
      <td>passed</td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2025.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>squeezenet1.1</td>
      <td>onnx</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>yolo26n</td>
      <td>pytorch</td>
      <td>FP32</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.3</td>
    </tr>
    <tr>
      <td>Image Classification</td>
      <td>squeezenet1.1</td>
      <td>onnx</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>yolo26n</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.3</td>
    </tr>
    <tr>
      <td>Object Detection</td>
      <td>yolo26n</td>
      <td>pytorch</td>
      <td>FP16-INT8</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td>passed</td>
      <td>passed</td>
      <td>OpenVINO 2026.3</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>bge-reranker-base</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.3</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>bge-reranker-base</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.3</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>bge-reranker-base</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td></td>
      <td>passed</td>
      <td></td>
      <td>OpenVINO 2026.3</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>exaone-4.0-1.2b</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td></td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.3</td>
    </tr>
    <tr>
      <td>Image Generation</td>
      <td>flux.2-klein-4b</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.3</td>
    </tr>
    <tr>
      <td>Image Generation</td>
      <td>flux.2-klein-4b</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.3</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>harrier-oss-v1-0.6b</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.3</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>harrier-oss-v1-0.6b</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.3</td>
    </tr>
    <tr>
      <td>Natural Language Processing</td>
      <td>harrier-oss-v1-0.6b</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.3</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>lfm2-1.2b</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.3</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>lfm2-1.2b</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.3</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>lfm2-1.2b</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.3</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>lfm2-24b-a2b</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.3</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>lfm2-24b-a2b</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.3</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>lfm2-8b-a1b</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.3</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>lfm2-8b-a1b</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.3</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>lfm2-8b-a1b</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.3</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>lfm2.5-1.2b-instruct</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.3</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>lfm2.5-1.2b-instruct</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.3</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>lfm2.5-1.2b-instruct</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.3</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>lfm2.5-350m</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.3</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>lfm2.5-350m</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.3</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>lfm2.5-350m</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.3</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>minicpm5-1b</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.3</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>minicpm5-1b</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.3</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>minicpm5-1b</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.3</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>nuextract-tiny</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.3</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>nuextract-tiny</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.3</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>nuextract-tiny</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.3</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>qwen3_8b_draft_eagle3</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.3</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>qwen3_8b_draft_eagle3</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.3</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>qwen3_8b_draft_eagle3</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.3</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>smollm3-3b</td>
      <td>pytorch</td>
      <td>FP16</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.3</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>smollm3-3b</td>
      <td>pytorch</td>
      <td>INT4-MIXED</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td>passed</td>
      <td>OpenVINO 2026.3</td>
    </tr>
    <tr>
      <td>Large Language Model</td>
      <td>smollm3-3b</td>
      <td>pytorch</td>
      <td>INT8-CW</td>
      <td>OpenVINO</td>
      <td>passed</td>
      <td></td>
      <td></td>
      <td>OpenVINO 2026.3</td>
    </tr>
  </tbody>
</table>

Marked cells indicate models that passed inference with no errors.  
  
In the precision column, the "optimum-intel default" label corresponds to FP32 for small models
and INT8 for models greater than 1B parameters.  
The results as of September 3rd 2025 for OpenVINO 2025.3; May 19th 2026 for OpenVINO 2026.2; July for OpenVINO 2026.3.  
The models come from different public model repositories, such as Pytorch Model Zoo and
HuggingFace; they were executed on the designated hardware with OpenVINO either natively or
as a backend.
