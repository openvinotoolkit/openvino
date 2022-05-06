# Converting TensorFlow EfficientDet Models {#openvino_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_EfficientDet_Models}

This tutorial explains how to convert EfficientDet public object detection models to the Intermediate Representation (IR).

## <a name="efficientdet-to-ir"></a>Converting EfficientDet Model to the IR

You will find on Github several public versions of EfficientDet model implementation. This tutorial explains how to
convert models from the [repository](https://github.com/google/automl/tree/master/efficientdet)
repository (commit 96e1fee) to the IR.

### Getting a Frozen TensorFlow Model

Follow the instructions below to get frozen TensorFlow EfficientDet model. EfficientDet-D4 model is an example:

1. Clone the repository:<br>
```sh
git clone https://github.com/google/automl
cd automl/efficientdet
```
2. (Optional) Checkout to the commit that the conversion was tested on:<br>
```sh
git checkout 96e1fee
```
3. Install required dependencies:<br>
```sh
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m pip install --upgrade tensorflow-model-optimization
```
4. Download and extract the model checkpoint [efficientdet-d4.tar.gz](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/efficientdet-d4.tar.gz)
referenced in the **"Pretrained EfficientDet Checkpoints"** section of the model repository:<br>
```sh
wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/efficientdet-d4.tar.gz
tar zxvf efficientdet-d4.tar.gz
```
5. Freeze the model:<br>
```sh
 mo --runmode=saved_model --model_name=efficientdet-d4  --ckpt_path=efficientdet-d4 --saved_model_dir=savedmodeldir
```
As a result, the frozen model file *`savedmodeldir/efficientdet-d4_frozen.pb`* will be generated.

> **NOTE**: For custom trained models, specify *`--hparams`* flag to *`config.yaml`* which was used during training.

> **NOTE**: If you see an error *`AttributeError: module 'tensorflow_core.python.keras.api._v2.keras.initializers' has no attribute 'variance_scaling'`*, apply the fix from the [patch](https://github.com/google/automl/pull/846).

### Converting an EfficientDet TensorFlow Model to the IR

To generate the IR of the EfficientDet TensorFlow model, run:<br>
```sh
mo \
--input_model savedmodeldir/efficientdet-d4_frozen.pb \
--transformations_config front/tf/automl_efficientdet.json \
--input_shape [1,$IMAGE_SIZE,$IMAGE_SIZE,3] \
--reverse_input_channels
```

Where *`$IMAGE_SIZE`* is the size that the input image of the original TensorFlow model will be resized to. Different
EfficientDet models were trained with different input image sizes. To determine the right one, refer to the *`efficientdet_model_param_dict`*
dictionary in the [hparams_config.py](https://github.com/google/automl/blob/96e1fee/efficientdet/hparams_config.py#L304) file.
The attribute *`image_size`* specifies the shape to be defined for the model conversion.

The *`transformations_config`* command line parameter specifies the configuration json file containing hints
for the Model Optimizer on how to convert the model and trigger transformations implemented in the
*`<PYTHON_SITE_PACKAGES>/openvino/tools/mo/front/tf/AutomlEfficientDet.py`*. The json file contains some parameters which must be changed if you
train the model yourself and modified the *`hparams_config`* file or the parameters are different from the ones used for EfficientDet-D4.
The attribute names are self-explanatory or match the name in the *`hparams_config`* file.

> **NOTE**: The color channel order (RGB or BGR) of an input data should match the channel order of the model training dataset. If they are different, perform the *`RGB<->BGR`* conversion specifying the command-line parameter: *`--reverse_input_channels`*. Otherwise, inference results may be incorrect. For more information about the parameter, refer to the **When to Reverse Input Channels** section of the [Converting a Model to Intermediate Representation (IR)](../Converting_Model.md) guide.

OpenVINO&trade; toolkit provides samples that can be used to infer EfficientDet model. 
For more information, refer to the [Open Model Zoo Demos](@ref omz_demos).

## <a name="efficientdet-ir-results-interpretation"></a>Interpreting Results of the TensorFlow Model and the IR

The TensorFlow model produces as output a list of 7-element tuples: *`[image_id, y_min, x_min, y_max, x_max, confidence, class_id]`*, where:
* *`image_id`* -- image batch index.
* *`y_min`* -- absolute *`y`* coordinate of the lower left corner of the detected object.
* *`x_min`* -- absolute *`x`* coordinate of the lower left corner of the detected object.
* *`y_max`* -- absolute *`y`* coordinate of the upper right corner of the detected object.
* *`x_max`* -- absolute *`x`* coordinate of the upper right corner of the detected object.
* *`confidence`* -- is the confidence of the detected object.
* *`class_id`* -- is the id of the detected object class counted from 1.

The output of the IR is a list of 7-element tuples: *`[image_id, class_id, confidence, x_min, y_min, x_max, y_max]`*, where:
* *`image_id`* -- image batch index.
* *`class_id`* -- is the id of the detected object class counted from 0.
* *`confidence`* -- is the confidence of the detected object.
* *`x_min`* -- normalized *`x`* coordinate of the lower left corner of the detected object.
* *`y_min`* -- normalized *`y`* coordinate of the lower left corner of the detected object.
* *`x_max`* -- normalized *`x`* coordinate of the upper right corner of the detected object.
* *`y_max`* -- normalized *`y`* coordinate of the upper right corner of the detected object.

The first element with *`image_id = -1`* means end of data.