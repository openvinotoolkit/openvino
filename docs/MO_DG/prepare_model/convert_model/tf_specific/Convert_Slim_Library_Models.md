# Converting TensorFlow Slim Image Classification Model Library Models {#openvino_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_Slim_Library_Models}

<a href="https://github.com/tensorflow/models/tree/master/research/slim/README.md">TensorFlow-Slim Image Classification Model Library</a> is a library to define, train and evaluate classification models in TensorFlow. The library contains Python scripts defining the classification topologies together with checkpoint files for several pre-trained classification topologies. To convert a TensorFlow-Slim library model, complete the following steps:

1. Download the TensorFlow-Slim models [git repository](https://github.com/tensorflow/models).
2. Download the pre-trained model [checkpoint](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models).
3. Export the inference graph.
4. Convert the model using the Model Optimizer.

The [Example of an Inception V1 Model Conversion](#example_of_an_inception_v1_model_conversion) below illustrates the process of converting an Inception V1 Model.

## Example of an Inception V1 Model Conversion <a name="example_of_an_inception_v1_model_conversion"></a>
This example demonstrates how to convert the model on Linux OSes, but it could be easily adopted for the Windows OSes.

**Step 1**. Create a new directory to clone the TensorFlow-Slim git repository to:

```sh
mkdir tf_models
```
```sh
git clone https://github.com/tensorflow/models.git tf_models
```

**Step 2**. Download and unpack the [Inception V1 model checkpoint file](http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz):

```sh
wget http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz
```
```sh
tar xzvf inception_v1_2016_08_28.tar.gz
```

**Step 3**. Export the inference graph --- the protobuf file (`.pb`) containing the architecture of the topology. This file *does not* contain the neural network weights and cannot be used for inference.

```sh
python3 tf_models/research/slim/export_inference_graph.py \
    --model_name inception_v1 \
    --output_file inception_v1_inference_graph.pb
```

Model Optimizer comes with the summarize graph utility, which identifies graph input and output nodes. Run the utility to determine input/output nodes of the Inception V1 model:

```sh
python3 <PYTHON_SITE_PACKAGES>/openvino/tools/mo/utils/summarize_graph.py --input_model ./inception_v1_inference_graph.pb
```

The output looks as follows:<br>
```sh
1 input(s) detected:
Name: input, type: float32, shape: (-1,224,224,3)
1 output(s) detected:
InceptionV1/Logits/Predictions/Reshape_1
```
The tool finds one input node with name `input`, type `float32`, fixed image size `(224,224,3)` and undefined batch size `-1`. The output node name is `InceptionV1/Logits/Predictions/Reshape_1`.<br>

**Step 4**. Convert the model with the Model Optimizer:

```sh
mo --input_model ./inception_v1_inference_graph.pb --input_checkpoint ./inception_v1.ckpt -b 1 --mean_value [127.5,127.5,127.5] --scale 127.5
```

The `-b` command line parameter is required because the Model Optimizer cannot convert a model with undefined input size.

For the information on why `--mean_values` and `--scale` command-line parameters are used, refer to the [Mean and Scale Values for TensorFlow-Slim Models](#tf_slim_mean_scale_values).

## Mean and Scale Values for TensorFlow-Slim Models <a name="tf_slim_mean_scale_values"></a>
The TensorFlow-Slim Models were trained with normalized input data. There are several different normalization algorithms used in the Slim library. OpenVINO classification sample does not perform image preprocessing except resizing to the input layer size. It is necessary to pass mean and scale values to the Model Optimizer so they are embedded into the generated IR in order to get correct classification results.

The file [preprocessing_factory.py](https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/preprocessing_factory.py) contains a dictionary variable `preprocessing_fn_map` defining mapping between the model type and preprocessing function to be used. The function code should be analyzed to figure out the mean/scale values.

The [inception_preprocessing.py](https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/inception_preprocessing.py) file defines the preprocessing function for the Inception models. The `preprocess_for_eval` function contains the following code:

```python3
    ...
    import tensorflow as tf
    if image.dtype != tf.float32:
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    ...
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image
```

Firstly, the `image` is converted to data type `tf.float32` and the values in the tensor are scaled to the `[0, 1]` range using the [tf.image.convert_image_dtype](https://www.tensorflow.org/api_docs/python/tf/image/convert_image_dtype) function. Then the `0.5` is subtracted from the image values and values multiplied by `2.0`. The final image range of values is `[-1, 1]`.

OpenVINO classification sample reads an input image as a three-dimensional array of integer values from the range `[0, 255]`. In order to scale them to `[-1, 1]` range, the mean value `127.5` for each image channel should be specified as well as a scale factor `127.5`.

Similarly, the mean/scale values can be determined for other Slim models.

The exact mean/scale values are defined in the table with list of supported TensorFlow-Slim models at the [Converting a TensorFlow Model](../Convert_Model_From_TensorFlow.md) guide.
