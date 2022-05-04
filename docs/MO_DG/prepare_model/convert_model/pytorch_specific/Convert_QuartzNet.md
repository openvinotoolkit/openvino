# Converting a PyTorch QuartzNet Model {#openvino_docs_MO_DG_prepare_model_convert_model_pytorch_specific_Convert_QuartzNet}

[NeMo project](https://github.com/NVIDIA/NeMo) provides the QuartzNet model.

## Downloading the Pretrained QuartzNet Model

To download the pretrained model, refer to the [NeMo Speech Models Catalog](https://ngc.nvidia.com/catalog/models/nvidia:nemospeechmodels).
Here are the instructions on how to obtain QuartzNet in ONNX format.

1. Install the NeMo toolkit, using the [instructions](https://github.com/NVIDIA/NeMo/tree/main#installation).

2. Run the following code:

```python
import nemo
import nemo.collections.asr as nemo_asr

quartznet = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="QuartzNet15x5Base-En")
# Export QuartzNet model to ONNX format
quartznet.decoder.export('decoder_qn.onnx')
quartznet.encoder.export('encoder_qn.onnx')
quartznet.export('qn.onnx')
```

This code produces 3 ONNX model files: *`encoder_qn.onnx`*, *`decoder_qn.onnx`*, *`qn.onnx`*.
They are *`decoder`*, *`encoder`*, and a combined *`decoder(encoder(x))`* models, respectively.

## Converting an ONNX QuartzNet model to IR

If using a combined model:
```sh
mo --input_model <MODEL_DIR>/qt.onnx --input_shape [B,64,X]
```
If using separate models:
```sh
mo --input_model <MODEL_DIR>/encoder_qt.onnx --input_shape [B,64,X]
mo --input_model <MODEL_DIR>/decoder_qt.onnx --input_shape [B,1024,Y]
```

Where shape is determined by the audio file Mel-Spectrogram length: B - batch dimension, X - dimension based on the input length, Y - determined by encoder output, usually *`X / 2`*.
