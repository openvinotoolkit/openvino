# Converting a PyTorch Cascade RCNN R-101 Model {#openvino_docs_MO_DG_prepare_model_convert_model_pytorch_specific_Convert_Cascade_RCNN_res101}

## Downloading and Converting a Model to ONNX

* Clone the [repository](https://github.com/open-mmlab/mmdetection):

```bash
git clone https://github.com/open-mmlab/mmdetection
cd mmdetection
```

> **NOTE**: To set up an environment, refer to the [instructions](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md#installation).

* Download the pretrained [model](https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_r101_fpn_1x_coco/cascade_rcnn_r101_fpn_1x_coco_20200317-0b6a2fbf.pth). The model is also available [here](https://github.com/open-mmlab/mmdetection/blob/master/configs/cascade_rcnn/README.md).

* To convert the model to ONNX format, use this [script](https://github.com/open-mmlab/mmdetection/blob/master/tools/deployment/pytorch2onnx.py).

```bash
python3 tools/deployment/pytorch2onnx.py configs/cascade_rcnn/cascade_rcnn_r101_fpn_1x_coco.py cascade_rcnn_r101_fpn_1x_coco_20200317-0b6a2fbf.pth --output-file cascade_rcnn_r101_fpn_1x_coco.onnx
```

The script generates ONNX model file *`cascade_rcnn_r101_fpn_1x_coco.onnx`* in the directory *`tools/deployment/`*. If required, specify the model name or output directory, using *`--output-file <path-to-dir>/<model-name>.onnx`* 

## Converting an ONNX Cascade RCNN R-101 Model to IR

```bash
mo --input_model cascade_rcnn_r101_fpn_1x_coco.onnx --mean_values [123.675,116.28,103.53] --scale_values [58.395,57.12,57.375]
```