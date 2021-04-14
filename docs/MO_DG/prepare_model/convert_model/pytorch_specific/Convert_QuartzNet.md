# Convert PyTorch* QuartzNet to the Intermediate Representation {#openvino_docs_MO_DG_prepare_model_convert_model_onnx_specific_Convert_QuartzNet}

[NeMo project](https://github.com/NVIDIA/NeMo) provides the QuartzNet model.

## Download the Pre-Trained QuartzNet Model

To download the pre-trained model, refer to the [NeMo Speech Models Catalog](https://ngc.nvidia.com/catalog/models/nvidia:nemospeechmodels).
Here are the instructions on how to obtain QuartzNet in ONNX* format.
```python
import nemo
import nemo.collections.asr as nemo_asr

quartznet = nemo_asr.models.ASRConvCTCModel.from_pretrained(model_info='QuartzNet15x5-En')
# Export QuartzNet model to ONNX* format
quartznet.export('qn.onnx')
```
This code produces 3 ONNX* model files: `encoder_qt.onnx`, `decoder_qt.onnx`, `qn.onnx`.
They are `decoder`, `encoder` and a combined `decoder(encoder(x))` models, respectively.

## Convert ONNX* QuartzNet model to IR

If using a combined model:
```sh
./mo.py --input_model <MODEL_DIR>/qt.onnx --input_shape [B,64,X]
```
If using separate models:
```sh
./mo.py --input_model <MODEL_DIR>/encoder_qt.onnx --input_shape [B,64,X]
./mo.py --input_model <MODEL_DIR>/decoder_qt.onnx --input_shape [B,1024,Y]
```

Where shape is determined by the audio file Mel-Spectrogram length: B - batch dimension, X - dimension based on the input length, Y - determined by encoder output, usually `X / 2`.
