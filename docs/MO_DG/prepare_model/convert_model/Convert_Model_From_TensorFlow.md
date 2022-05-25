# Converting a TensorFlow Model {#openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow}

## Converting TensorFlow 1 Models <a name="Convert_From_TF2X"></a>

### Converting Frozen Model Format <a name="Convert_From_TF"></a>
To convert a TensorFlow model, use the *`mo`* script to simply convert a model with a path to the input model *`.pb`* file:

```sh
 mo --input_model <INPUT_MODEL>.pb
```

### Converting Non-Frozen Model Formats <a name="loading-nonfrozen-models"></a>
There are three ways to store non-frozen TensorFlow models and convert them by Model Optimizer:

1. **Checkpoint**. In this case, a model consists of two files: `inference_graph.pb` (or `inference_graph.pbtxt`) and `checkpoint_file.ckpt`.
If you do not have an inference graph file, refer to the [Freezing Custom Models in Python](#freeze-the-tensorflow-model) section.
To convert the model with the inference graph in `.pb` format, run the `mo` script with a path to the checkpoint file:
```sh
 mo --input_model <INFERENCE_GRAPH>.pb --input_checkpoint <INPUT_CHECKPOINT>
```
To convert the model with the inference graph in `.pbtxt` format, run the `mo` script with a path to the checkpoint file:
```sh
 mo --input_model <INFERENCE_GRAPH>.pbtxt --input_checkpoint <INPUT_CHECKPOINT> --input_model_is_text
```

2. **MetaGraph**. In this case, a model consists of three or four files stored in the same directory: `model_name.meta`, `model_name.index`,
`model_name.data-00000-of-00001` (the numbers may vary), and `checkpoint` (optional).
To convert such TensorFlow model, run the `mo` script with a path to the MetaGraph `.meta` file:
```sh
 mo --input_meta_graph <INPUT_META_GRAPH>.meta
```

3. **SavedModel format**. In this case, a model consists of a special directory with a `.pb` file
and several subfolders: `variables`, `assets`, and `assets.extra`. For more information about the SavedModel directory, refer to the [README](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python/saved_model#components) file in the TensorFlow repository.
To convert such TensorFlow model, run the `mo` script with a path to the SavedModel directory:
```sh
 mo --saved_model_dir <SAVED_MODEL_DIRECTORY>
```

You can convert TensorFlow 1.x SavedModel format in the environment that has a 1.x or 2.x version of TensorFlow. However, TensorFlow 2.x SavedModel format strictly requires the 2.x version of TensorFlow.
If a model contains operations currently unsupported by OpenVINO, prune these operations by explicit specification of input nodes using the `--input` option.
To determine custom input nodes, display a graph of the model in TensorBoard. To generate TensorBoard logs of the graph, use the `--tensorboard_logs` option.
TensorFlow 2.x SavedModel format has a specific graph due to eager execution. In case of pruning, find custom input nodes in the `StatefulPartitionedCall/*` subgraph of TensorFlow 2.x SavedModel format.

### Freezing Custom Models in Python <a name="freeze-the-tensorflow-model"></a>
When a network is defined in Python code, you have to create an inference graph file. Graphs are usually built in a form
that allows model training. That means all trainable parameters are represented as variables in the graph.
To be able to use such graph with Model Optimizer, it should be frozen and dumped to a file with the following code:

```python
import tensorflow as tf
from tensorflow.python.framework import graph_io
frozen = tf.compat.v1.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["name_of_the_output_node"])
graph_io.write_graph(frozen, './', 'inference_graph.pb', as_text=False)
```

Where:

* `sess` is the instance of the TensorFlow Session object where the network topology is defined.
* `["name_of_the_output_node"]` is the list of output node names in the graph; `frozen` graph will
    include only those nodes from the original `sess.graph_def` that are directly or indirectly used
    to compute given output nodes. The `'name_of_the_output_node'` is an example of a possible output
    node name. You should derive the names based on your own graph.
* `./` is the directory where the inference graph file should be generated.
* `inference_graph.pb` is the name of the generated inference graph file.
* `as_text` specifies whether the generated file should be in human readable text format or binary.

## Converting TensorFlow 2 Models <a name="Convert_From_TF2X"></a>
To convert TensorFlow 2 models, ensure that `openvino-dev[tensorflow2]` is installed via `pip`.
TensorFlow 2.X officially supports two model formats: SavedModel and Keras H5 (or HDF5).
Below are the instructions on how to convert each of them.

### SavedModel Format
A model in the SavedModel format consists of a directory with a `saved_model.pb` file and two subfolders: `variables` and `assets`.
To convert such a model, run the `mo` script with a path to the SavedModel directory:

```sh
 mo --saved_model_dir <SAVED_MODEL_DIRECTORY>
```

TensorFlow 2 SavedModel format strictly requires the 2.x version of TensorFlow installed in the
environment for conversion to the Intermediate Representation (IR).

If a model contains operations currently unsupported by OpenVINO™,
prune these operations by explicit specification of input nodes using the `--input` or `--output`
options. To determine custom input nodes, visualize a model graph in the TensorBoard.

To generate TensorBoard logs of the graph, use the Model Optimizer `--tensorboard_logs` command-line
option.

TensorFlow 2 SavedModel format has a specific graph structure due to eager execution. In case of
pruning, find custom input nodes in the `StatefulPartitionedCall/*` subgraph.

### Keras H5
If you have a model in the HDF5 format, load the model using TensorFlow 2 and serialize it in the
SavedModel format. Here is an example of how to do it:

```python
import tensorflow as tf
model = tf.keras.models.load_model('model.h5')
tf.saved_model.save(model,'model')
```

The Keras H5 model with a custom layer has specifics to be converted into SavedModel format.
For example, the model with a custom layer `CustomLayer` from `custom_layer.py` is converted as follows:

```python
import tensorflow as tf
from custom_layer import CustomLayer
model = tf.keras.models.load_model('model.h5', custom_objects={'CustomLayer': CustomLayer})
tf.saved_model.save(model,'model')
```

Then follow the above instructions for the SavedModel format.

> **NOTE**: Do not use other hacks to resave TensorFlow 2 models into TensorFlow 1 formats.

## Command-Line Interface (CLI) Examples Using TensorFlow-Specific Parameters
* Launching the Model Optimizer for Inception V1 frozen model when model file is a plain text protobuf:

```sh
 mo --input_model inception_v1.pbtxt --input_model_is_text -b 1
```

* Launching the Model Optimizer for Inception V1 frozen model and dump information about the graph to TensorBoard log dir `/tmp/log_dir`

```sh
 mo --input_model inception_v1.pb -b 1 --tensorboard_logdir /tmp/log_dir
```

* Launching the Model Optimizer for BERT model in the SavedModel format, with three inputs. Specify explicitly the input shapes
where the batch size and the sequence length equal 2 and 30 respectively.

```sh
mo --saved_model_dir BERT --input mask,word_ids,type_ids --input_shape [2,30],[2,30],[2,30]
```

## Supported TensorFlow and TensorFlow 2 Keras Layers
For the list of supported standard layers, refer to the [Supported Framework Layers ](../Supported_Frameworks_Layers.md) page.

## Frequently Asked Questions (FAQ)
The Model Optimizer provides explanatory messages if it is unable to run to completion due to typographical errors, incorrectly used options, or other issues. The message describes the potential cause of the problem and gives a link to the [Model Optimizer FAQ](../Model_Optimizer_FAQ.md). The FAQ provides instructions on how to resolve most issues. The FAQ also includes links to relevant sections in the Model Optimizer Developer Guide to help you understand what went wrong.

## Summary
In this document, you learned:

* Basic information about how the Model Optimizer works with TensorFlow models.
* Which TensorFlow models are supported.
* How to freeze a TensorFlow model.
* How to convert a trained TensorFlow model using the Model Optimizer with both framework-agnostic and TensorFlow-specific command-line options.

## See Also
[Model Conversion Tutorials](Convert_Model_Tutorials.md)
