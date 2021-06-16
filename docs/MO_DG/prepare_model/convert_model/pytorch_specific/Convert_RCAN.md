# Convert PyTorch* RCAN to the Intermediate Representation {#openvino_docs_MO_DG_prepare_model_convert_model_pytorch_specific_Convert_RCAN}

[RCAN](https://github.com/yulunzhang/RCAN): Image Super-Resolution Using Very Deep Residual Channel Attention Networks

## Download and Convert the Model to ONNX*

To download the pre-trained model or train the model yourself, refer to the 
[instruction](https://github.com/yulunzhang/RCAN/blob/master/README.md) in the RCAN model repository. Firstly, 
convert the model to ONNX\* format. :

```sh
python main.py --scale 2 --model RCAN --n_resgroups 10 --n_resblocks 20 --n_feats 64 --pre_train ../model/RCAN_BIX2.pt --chop
```

The script generates the ONNX\* model file RCAN.onnx. 

## Convert ONNX* RCAN Model to IR

```sh
./mo.py --input_model RCAN.onnx
```
