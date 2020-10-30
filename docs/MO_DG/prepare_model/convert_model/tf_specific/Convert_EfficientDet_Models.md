# Converting EfficientDet Models from TensorFlow {#openvino_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_EfficientDet_Models}

This tutorial explains how to convert detection EfficientDet\* public models to the Intermediate Representation (IR). 

## <a name="efficientdet-to-ir"></a>Convert EfficientDet Model to IR

On GitHub*, you can find several public versions of EfficientDet model implementation. This tutorial explains how to 
convert models from the [https://github.com/google/automl/tree/master/efficientdet](https://github.com/google/automl/tree/master/efficientdet) 
repository (commit 96e1fee) to IR.

### Get Frozen TensorFlow\* Model

Follow the instructions below to get frozen TensorFlow EfficientDet model. We use EfficientDet-D4 model as an example:

1. Clone the repository:<br>
```sh
git clone https://github.com/google/automl
cd automl/efficientdet
```
2. (Optional) Checkout to the commit that the conversion was tested on:<br>
```sh
git checkout 96e1fee
```
3. Download and extract the model checkpoint [efficientdet-d4.tar.gz](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/efficientdet-d4.tar.gz)
referenced in the "Pretrained EfficientDet Checkpoints" section of the model repository:<br>
```sh
wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/efficientdet-d4.tar.gz
tar zxvf efficientdet-d4.tar.gz
```
4. Freeze the model:<br>
```sh
python3 model_inspect.py --runmode=saved_model --model_name=efficientdet-d4  --ckpt_path=efficientdet-d4 --saved_model_dir=savedmodeldir
```
As a result the frozen model file `savedmodeldir/efficientdet-d4_frozen.pb` will be generated.

### Convert EfficientDet TensorFlow Model to the IR

To generate the IR of the EfficientDet TensorFlow model, run:<br>
```sh
python3 $MO_ROOT/mo.py \
--input_model savedmodeldir/efficientdet-d4_frozen.pb \
--tensorflow_use_custom_operations_config $MO_ROOT/extensions/front/tf/automl_efficientdet.json \
--input_shape [1,$IMAGE_SIZE,$IMAGE_SIZE,3] \
--reverse_input_channels
```

Where `$IMAGE_SIZE` is the size that the input image of the original TensorFlow model will be resized to. Different
EfficientDet models were trained with different input image sizes. To determine the right one refer to the `efficientdet_model_param_dict`
dictionary in the [hparams_config.py](https://github.com/google/automl/blob/96e1fee/efficientdet/hparams_config.py#L304) file.
The attribute `image_size` specifies the shape to be specified for the model conversion.

The `tensorflow_use_custom_operations_config` command line parameter specifies the configuration json file containing hints
to the Model Optimizer on how to convert the model and trigger transformations implemented in the 
`$MO_ROOT/extensions/front/tf/AutomlEfficientDet.py`. The json file contains some parameters which must be changed if you
train the model yourself and modified the `hparams_config` file or the parameters are different from the ones used for EfficientDet-D4.
The attribute names are self-explanatory or match the name in the `hparams_config` file.

> **NOTE:** The color channel order (RGB or BGR) of an input data should match the channel order of the model training dataset. If they are different, perform the `RGB<->BGR` conversion specifying the command-line parameter: `--reverse_input_channels`. Otherwise, inference results may be incorrect. For more information about the parameter, refer to **When to Reverse Input Channels** section of [Converting a Model Using General Conversion Parameters](../Converting_Model_General.md).

OpenVINO&trade; toolkit provides samples that can be used to infer EfficientDet model. For more information, refer to 
[Object Detection for SSD C++ Sample](@ref openvino_inference_engine_samples_object_detection_sample_ssd_README) and 
[Object Detection for SSD Python Sample](@ref openvino_inference_engine_ie_bridges_python_sample_object_detection_sample_ssd_README).

---
## See Also

* [Sub-Graph Replacement in Model Optimizer](../../customize_model_optimizer/Subgraph_Replacement_Model_Optimizer.md)
