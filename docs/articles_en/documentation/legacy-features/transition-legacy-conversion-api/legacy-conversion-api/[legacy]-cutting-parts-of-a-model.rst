[LEGACY] Cutting Off Parts of a Model
================================================

.. danger::

   The code described here has been **deprecated!** Do not use it to avoid working with a legacy solution. It will be kept for some time to ensure backwards compatibility, but **you should not use** it in contemporary applications.

Sometimes, it is necessary to remove parts of a model when converting it to OpenVINO IR. This chapter describes how to do it, using model conversion API parameters. Model cutting applies mostly to TensorFlow models, which is why TensorFlow will be used in this chapter's examples, but it may be also useful for other frameworks.

Purpose of Model Cutting
########################

The following examples are the situations when model cutting is useful or even required:

* A model has pre- or post-processing parts that cannot be translated to existing OpenVINO operations.
* A model has a training part that is convenient to be kept in the model but not used during inference.
* A model is too complex be converted at once, because it contains a lot of unsupported operations that cannot be easily implemented as custom layers.
* A problem occurs with model conversion or inference in OpenVINO™ Runtime. To identify the issue, limit the conversion scope by iterative search for problematic areas in the model.
* A single custom layer or a combination of custom layers is isolated for debugging purposes.

.. note::

   Internally, when you run model conversion API, it loads the model, goes through the topology, and tries to find each layer type in a list of known layers. Custom layers are layers that are not included in the list. If your topology contains such kind of layers, model conversion API classifies them as custom.

Model conversion API parameters
###############################

Model conversion API provides ``input`` and ``output`` command-line options to specify new entry and exit nodes, while ignoring the rest of the model:

* ``input`` option accepts a list of layer names of the input model that should be treated as new entry points to the model. See the full list of accepted types for input on :doc:`Model Conversion Python API <[legacy]-convert-models-as-python-objects>` page.
* ``output`` option accepts a list of layer names of the input model that should be treated as new exit points from the model.

The ``input`` option is required for cases unrelated to model cutting. For example, when the model contains several inputs and ``input_shape`` or ``mean_values`` options are used, the ``input`` option specifies the order of input nodes for correct mapping between multiple items provided in ``input_shape`` and ``mean_values`` and the inputs in the model.

Model cutting is illustrated with the Inception V1 model, found in the ``models/research/slim`` repository. To proceed with this chapter, make sure you do the necessary steps to :doc:`prepare the model for model conversion <[legacy]-setting-input-shapes>`.

Default Behavior without input and output
#########################################

The input model is converted as a whole if neither ``input`` nor ``output`` command line options are used. All ``Placeholder`` operations in a TensorFlow graph are automatically identified as entry points. The ``Input`` layer type is generated for each of them. All nodes that have no consumers are automatically identified as exit points.

For Inception_V1, there is one ``Placeholder``: input. If the model is viewed in TensorBoard, the input operation is easy to find:

.. image:: ../../../../assets/images/inception_v1_std_input.svg
   :alt: Placeholder in Inception V1

``Reshape`` is the only output operation, which is enclosed in a nested name scope of ``InceptionV1/Logits/Predictions``, under the full name of ``InceptionV1/Logits/Predictions/Reshape_1``.

In TensorBoard, along with some of its predecessors, it looks as follows:

.. image:: ../../../../assets/images/inception_v1_std_output.svg
   :alt: TensorBoard with predecessors

Convert this model to ``ov.Model``:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. code-block:: py
         :force:

         from openvino.tools.mo import convert_model
         ov_model = convert_model("inception_v1.pb", batch=1)

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: sh

         mo --input_model inception_v1.pb -b 1 --output_dir <OUTPUT_MODEL_DIR>


``ov.Model`` can be serialized with the ``ov.serialize()`` method to Intermediate Representation which can be used for model structure exploring.
In IR, the structure of a model has the following layers:

.. code-block:: xml
   :force:

   <layer id="286" name="input" precision="FP32" type="Input">
       <output>
           <port id="0">
               <dim>1</dim>
               <dim>3</dim>
               <dim>224</dim>
               <dim>224</dim>
           </port>
       </output>
   </layer>


The ``input`` layer is converted from the TensorFlow graph ``Placeholder`` operation ``input`` and has the same name.

The ``-b`` option is used here for conversion to override a possible undefined batch size (coded as -1 in TensorFlow models). If a model was frozen with a defined batch size, you may omit this option in all the examples.

The last layer in the model is ``InceptionV1/Logits/Predictions/Reshape_1``, which matches an output operation in the TensorFlow graph:

.. code-block:: xml
   :force:

   <layer id="389" name="InceptionV1/Logits/Predictions/Reshape_1" precision="FP32" type="Reshape">
       <data axis="0" dim="1,1001" num_axes="-1"/>
       <input>
           <port id="0">
               <dim>1</dim>
               <dim>1001</dim>
           </port>
       </input>
       <output>
           <port id="1">
               <dim>1</dim>
               <dim>1001</dim>
           </port>
       </output>
   </layer>


Due to automatic identification of inputs and outputs, providing the ``input`` and ``output`` options to convert the whole model is not required. The following commands are equivalent for the Inception V1 model:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. code-block:: py
         :force:

         from openvino.tools.mo import convert_model
         ov_model = convert_model("inception_v1.pb", batch=1)

         ov_model = convert_model("inception_v1.pb", batch=1, input="input", output="InceptionV1/Logits/Predictions/Reshape_1")

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: sh

         mo --input_model inception_v1.pb -b 1 --output_dir <OUTPUT_MODEL_DIR>

         mo --input_model inception_v1.pb -b 1 --input input --output InceptionV1/Logits/Predictions/Reshape_1 --output_dir <OUTPUT_MODEL_DIR>


The Intermediate Representations are identical for both conversions. The same is true if the model has multiple inputs and/or outputs.

Model Cutting
####################

Now, consider how to cut some parts of the model off. This chapter describes the first convolution block ``InceptionV1/InceptionV1/Conv2d_1a_7x7`` of the Inception V1 model to illustrate cutting:

.. image:: ../../../../assets/images/inception_v1_first_block.svg
   :alt: Inception V1 first convolution block

Cutting at the End
++++++++++++++++++++

If you want to cut your model at the end, you have the following options:

1. The following command cuts off the rest of the model after the ``InceptionV1/InceptionV1/Conv2d_1a_7x7/Relu``, making this node the last in the model:

   .. tab-set::

      .. tab-item:: Python
         :sync: py

         .. code-block:: py
            :force:

            from openvino.tools.mo import convert_model
            ov_model = convert_model("inception_v1.pb", batch=1, output="InceptionV1/InceptionV1/Conv2d_1a_7x7/Relu")

      .. tab-item:: CLI
         :sync: cli

         .. code-block:: sh

            mo --input_model inception_v1.pb -b 1 --output=InceptionV1/InceptionV1/Conv2d_1a_7x7/Relu --output_dir <OUTPUT_MODEL_DIR>


   The resulting Intermediate Representation has three layers:

   .. code-block:: xml
      :force:

      <?xml version="1.0" ?>
      <net batch="1" name="model" version="2">
         <layers>
            <layer id="3" name="input" precision="FP32" type="Input">
               <output>
                  <port id="0">...</port>
               </output>
            </layer>
            <layer id="5" name="InceptionV1/InceptionV1/Conv2d_1a_7x7/convolution" precision="FP32" type="Convolution">
               <data dilation-x="1" dilation-y="1" group="1" kernel-x="7" kernel-y="7" output="64" pad-x="2" pad-y="2" stride="1,1,2,2" stride-x="2" stride-y="2"/>
               <input>
                  <port id="0">...</port>
               </input>
               <output>
                  <port id="3">...</port>
               </output>
               <blobs>
                  <weights offset="0" size="37632"/>
                  <biases offset="37632" size="256"/>
               </blobs>
            </layer>
            <layer id="6" name="InceptionV1/InceptionV1/Conv2d_1a_7x7/Relu" precision="FP32" type="ReLU">
               <input>
                  <port id="0">...</port>
               </input>
               <output>
                  <port id="1">...</port>
               </output>
            </layer>
         </layers>
         <edges>
            <edge from-layer="3" from-port="0" to-layer="5" to-port="0"/>
            <edge from-layer="5" from-port="3" to-layer="6" to-port="0"/>
         </edges>
      </net>


   As shown in the TensorBoard picture, the original model has more nodes than its Intermediate Representation. Model conversion, using ``convert_model()``, consists of a set of model transformations, including fusing of batch normalization ``InceptionV1/InceptionV1/Conv2d_1a_7x7/BatchNorm`` with convolution ``InceptionV1/InceptionV1/Conv2d_1a_7x7/convolution``, which is why it is not present in the final model. This is not an effect of the ``output`` option, it is the typical behavior of model conversion API for batch normalizations and convolutions. The effect of the ``output`` is that the ``ReLU`` layer becomes the last one in the converted model.

2. The following command cuts the edge that comes from 0 output port of the ``InceptionV1/InceptionV1/Conv2d_1a_7x7/Relu`` and the rest of the model, making this node the last one in the model:

   .. tab-set::

      .. tab-item:: Python
         :sync: py

         .. code-block:: py
            :force:

            from openvino.tools.mo import convert_model
            ov_model = convert_model("inception_v1.pb", batch=1, output="InceptionV1/InceptionV1/Conv2d_1a_7x7/Relu:0")

      .. tab-item:: CLI
         :sync: cli

         .. code-block:: sh

            mo --input_model inception_v1.pb -b 1 --output InceptionV1/InceptionV1/Conv2d_1a_7x7/Relu:0 --output_dir <OUTPUT_MODEL_DIR>


   The resulting Intermediate Representation has three layers, which are the same as in the previous case:

   .. code-block:: xml
      :force:

      <?xml version="1.0" ?>
      <net batch="1" name="model" version="2">
         <layers>
            <layer id="3" name="input" precision="FP32" type="Input">
               <output>
                  <port id="0">...</port>
               </output>
            </layer>
            <layer id="5" name="InceptionV1/InceptionV1/Conv2d_1a_7x7/convolution" precision="FP32" type="Convolution">
               <data dilation-x="1" dilation-y="1" group="1" kernel-x="7" kernel-y="7" output="64" pad-x="2" pad-y="2" stride="1,1,2,2" stride-x="2" stride-y="2"/>
               <input>
                  <port id="0">...</port>
               </input>
               <output>
                  <port id="3">...</port>
               </output>
               <blobs>
                  <weights offset="0" size="37632"/>
                  <biases offset="37632" size="256"/>
               </blobs>
            </layer>
            <layer id="6" name="InceptionV1/InceptionV1/Conv2d_1a_7x7/Relu" precision="FP32" type="ReLU">
               <input>
                  <port id="0">...</port>
               </input>
               <output>
                  <port id="1">...</port>
               </output>
            </layer>
         </layers>
         <edges>
         	<edge from-layer="3" from-port="0" to-layer="5" to-port="0"/>
         	<edge from-layer="5" from-port="3" to-layer="6" to-port="0"/>
         </edges>
      </net>


   This type of cutting is useful for cutting multiple output edges.

3. The following command cuts the edge that comes to 0 input port of the ``InceptionV1/InceptionV1/Conv2d_1a_7x7/Relu`` and the rest of the model including ``InceptionV1/InceptionV1/Conv2d_1a_7x7/Relu``, deleting this node and making the previous node ``InceptionV1/InceptionV1/Conv2d_1a_7x7/Conv2D`` the last in the model:

   .. tab-set::

      .. tab-item:: Python
         :sync: py

         .. code-block:: py
            :force:

            from openvino.tools.mo import convert_model
            ov_model = convert_model("inception_v1.pb", batch=1, output="0:InceptionV1/InceptionV1/Conv2d_1a_7x7/Relu")

      .. tab-item:: CLI
         :sync: cli

         .. code-block:: sh

            mo --input_model inception_v1.pb -b 1 --output=0:InceptionV1/InceptionV1/Conv2d_1a_7x7/Relu --output_dir <OUTPUT_MODEL_DIR>


   The resulting Intermediate Representation has two layers, which are the same as the first two layers in the previous case:

   .. code-block:: xml
      :force:

      <?xml version="1.0" ?>
      <net batch="1" name="inception_v1" version="2">
         <layers>
            <layer id="0" name="input" precision="FP32" type="Input">
               <output>
                  <port id="0">...</port>
               </output>
            </layer>
            <layer id="1" name="InceptionV1/InceptionV1/Conv2d_1a_7x7/Conv2D" precision="FP32" type="Convolution">
               <data auto_pad="same_upper" dilation-x="1" dilation-y="1" group="1" kernel-x="7" kernel-y="7" output="64" pad-b="3" pad-r="3" pad-x="2" pad-y="2" stride="1,1,2,   2"       stride-x="2" stride-y="2"/>
               <input>
                  <port id="0">...</port>
               </input>
               <output>
                  <port id="3">...</port>
               </output>
               <blobs>
                  <weights offset="0" size="37632"/>
                  <biases offset="37632" size="256"/>
               </blobs>
            </layer>
         </layers>
         <edges>
            <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
         </edges>
      </net>


Cutting from the Beginning
++++++++++++++++++++++++++

If you want to go further and cut the beginning of the model, leaving only the ``ReLU`` layer, you have the following options:

1. Use the following parameters, where ``input`` and ``output`` specify the same node in the graph:

   .. tab-set::

      .. tab-item:: Python
         :sync: py

         .. code-block:: py
            :force:

            from openvino.tools.mo import convert_model
            ov_model = convert_model("inception_v1.pb", batch=1, output="InceptionV1/InceptionV1/Conv2d_1a_7x7/Relu", input="InceptionV1/InceptionV1/Conv2d_1a_7x7/Relu")

      .. tab-item:: CLI
         :sync: cli

         .. code-block:: sh

            mo --input_model=inception_v1.pb -b 1 --output InceptionV1/InceptionV1/Conv2d_1a_7x7/Relu --input InceptionV1/InceptionV1/Conv2d_1a_7x7/Relu --output_dir <OUTPUT_MODEL_DIR>


   The resulting Intermediate Representation looks as follows:

   .. code-block:: xml
      :force:

      <xml version="1.0">
      <net batch="1" name="model" version="2">
         <layers>
            <layer id="0" name="InceptionV1/InceptionV1/Conv2d_1a_7x7/Relu/placeholder_port_0" precision="FP32" type="Input">
               <output>
                  <port id="0">...</port>
               </output>
            </layer>
            <layer id="2" name="InceptionV1/InceptionV1/Conv2d_1a_7x7/Relu" precision="FP32" type="ReLU">
               <input>
                  <port id="0">...</port>
               </input>
               <output>
                  <port id="1">...</port>
               </output>
            </layer>
         </layers>
         <edges>
            <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
         </edges>
      </net>


   ``Input`` layer is automatically created to feed the layer that is converted from the node specified in ``input``, which is ``InceptionV1/InceptionV1/Conv2d_1a_7x7/Relu`` in this case. ``convert_model()`` does not replace the ``ReLU`` node by the ``Input`` layer. It produces such ``ov.Model`` to make the node the first executable node in the final Intermediate Representation. Therefore, model conversion creates enough ``Inputs`` to feed all input ports of the node that is passed in ``input``.

   Even though ``input_shape`` is not specified in the command line, the shapes for layers are inferred from the beginning of the original TensorFlow model to the point, at which the new input is defined. It has the same shape ``[1,64,112,112]`` as the model converted as a whole or without cutting off the beginning.

2. Cut the edge incoming to layer by port number. To specify the incoming port, use the following notation ``input=port:input_node``. To cut everything before ``ReLU`` layer, cut the edge incoming to port 0 of ``InceptionV1/InceptionV1/Conv2d_1a_7x7/Relu`` node:

   .. tab-set::

      .. tab-item:: Python
         :sync: py

         .. code-block:: py
            :force:

            from openvino.tools.mo import convert_model
            ov_model = convert_model("inception_v1.pb", batch=1, input="0:InceptionV1/InceptionV1/Conv2d_1a_7x7/Relu", output="InceptionV1/InceptionV1/Conv2d_1a_7x7/Relu")

      .. tab-item:: CLI
         :sync: cli

         .. code-block:: sh

            mo --input_model inception_v1.pb -b 1 --input 0:InceptionV1/InceptionV1/Conv2d_1a_7x7/Relu --output InceptionV1/InceptionV1/Conv2d_1a_7x7/Relu --output_dir <OUTPUT_MODEL_DIR>


   The resulting Intermediate Representation looks as follows:

   .. code-block:: xml
      :force:

      <xml version="1.0">
      <net batch="1" name="model" version="2">
         <layers>
            <layer id="0" name="InceptionV1/InceptionV1/Conv2d_1a_7x7/Relu/placeholder_port_0" precision="FP32" type="Input">
               <output>
                  <port id="0">...</port>
               </output>
            </layer>
            <layer id="2" name="InceptionV1/InceptionV1/Conv2d_1a_7x7/Relu" precision="FP32" type="ReLU">
               <input>
                  <port id="0">...</port>
               </input>
               <output>
                  <port id="1">...</port>
               </output>
            </layer>
         </layers>
         <edges>
            <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
         </edges>
      </net>


   ``Input`` layer is automatically created to feed the layer that is converted from the node specified in ``input``, which is ``InceptionV1/InceptionV1/Conv2d_1a_7x7/Relu`` in this case. ``convert_model()`` does not replace the ``ReLU`` node by the ``Input`` layer, it produces such ``ov.Model`` to make the node be the first executable node in the final Intermediate Representation. Therefore, ``convert_model()`` creates enough ``Inputs`` to feed all input ports of the node that is passed in ``input``.

   Even though ``input_shape`` is not specified in the command line, the shapes for layers are inferred from the beginning of the original TensorFlow model to the point, at which the new input is defined. It has the same shape ``[1,64,112,112]`` as the model converted as a whole or without cutting off the beginning.

3. Cut edge outcoming from layer by port number. To specify the outcoming port, use the following notation ``input=input_node:port``. To cut everything before ``ReLU`` layer, cut edge from ``InceptionV1/InceptionV1/Conv2d_1a_7x7/BatchNorm/batchnorm/add_1`` node to ``ReLU``:

   .. tab-set::

      .. tab-item:: Python
         :sync: py

         .. code-block:: py
            :force:

            from openvino.tools.mo import convert_model
            ov_model = convert_model("inception_v1.pb", batch=1, input="InceptionV1/InceptionV1/Conv2d_1a_7x7/BatchNorm/batchnorm/add_1:0", output="InceptionV1/InceptionV1/Conv2d_1a_7x7/Relu")

      .. tab-item:: CLI
         :sync: cli

         .. code-block:: sh

            mo --input_model inception_v1.pb -b 1 --input InceptionV1/InceptionV1/Conv2d_1a_7x7/BatchNorm/batchnorm/add_1:0 --output InceptionV1/InceptionV1/Conv2d_1a_7x7/Relu --output_dir <OUTPUT_MODEL_DIR>


   The resulting Intermediate Representation looks as follows:

   .. code-block:: xml
      :force:

      <xml version="1.0">
      <net batch="1" name="model" version="2">
         <layers>
            <layer id="0" name="InceptionV1/InceptionV1/Conv2d_1a_7x7/BatchNorm/batchnorm/add_1/placeholder_out_port_0" precision="FP32" type="Input">
               <output>
                  <port id="0">...</port>
               </output>
            </layer>
            <layer id="1" name="InceptionV1/InceptionV1/Conv2d_1a_7x7/Relu" precision="FP32" type="ReLU">
               <input>
                  <port id="0">...</port>
               </input>
               <output>
                  <port id="1">...</port>
               </output>
               layer>
         </layers>
         <edges>
            <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
         </edges>
      </net>


Inputs with Multiple Input Ports
################################

There are operations that contain more than one input port. In the example considered here, the convolution ``InceptionV1/InceptionV1/Conv2d_1a_7x7/convolution`` is such operation. When ``input_shape`` is not provided, a new ``Input`` layer is created for each dynamic input port for the node. If a port is evaluated to a constant blob, this constant remains in the model and a corresponding input layer is not created. TensorFlow convolution used in this model contains two ports:

* port 0: input tensor for convolution (dynamic)
* port 1: convolution weights (constant)

Following this behavior, ``convert_model()`` creates an ``Input`` layer for port 0 only, leaving port 1 as a constant. Thus, the result of:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. code-block:: py
         :force:

         from openvino.tools.mo import convert_model
         ov_model = convert_model("inception_v1.pb", batch=1, input="InceptionV1/InceptionV1/Conv2d_1a_7x7/convolution")

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: sh

         mo --input_model inception_v1.pb -b 1 --input InceptionV1/InceptionV1/Conv2d_1a_7x7/convolution --output_dir <OUTPUT_MODEL_DIR>


is identical to the result of conversion of the model as a whole, because this convolution is the first executable operation in Inception V1.

Different behavior occurs when ``input_shape`` is also used as an attempt to override the input shape:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. code-block:: py
         :force:

         from openvino.tools.mo import convert_model
         ov_model = convert_model("inception_v1.pb", input="InceptionV1/InceptionV1/Conv2d_1a_7x7/convolution", input_shape=[1,224,224,3])

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: sh

         mo --input_model inception_v1.pb--input=InceptionV1/InceptionV1/Conv2d_1a_7x7/convolution --input_shape [1,224,224,3]  --output_dir <OUTPUT_MODEL_DIR>


An error occurs (for more information, see the :ref:`Model Conversion FAQ <question-30>`):

.. code-block:: sh

   [ ERROR ]  Node InceptionV1/InceptionV1/Conv2d_1a_7x7/convolution has more than 1 input and input shapes were provided.
   Try not to provide input shapes or specify input port with PORT:NODE notation, where PORT is an integer.
   For more information, see FAQ #30

When ``input_shape`` is specified and the node contains multiple input ports, you need to provide an input port index together with an input node name. The input port index is specified in front of the node name with ``‘:’`` as a separator (``PORT:NODE``). In this case, the port index 0 of the node ``InceptionV1/InceptionV1/Conv2d_1a_7x7/convolution`` should be specified as ``0:InceptionV1/InceptionV1/Conv2d_1a_7x7/convolution``.

The correct command line is:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. code-block:: py
         :force:

         from openvino.tools.mo import convert_model
         ov_model = convert_model("inception_v1.pb", input="0:InceptionV1/InceptionV1/Conv2d_1a_7x7/convolution", input_shape=[1,224,224,3])

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: sh

         mo --input_model inception_v1.pb --input 0:InceptionV1/InceptionV1/Conv2d_1a_7x7/convolution --input_shape=[1,224,224,3] --output_dir <OUTPUT_MODEL_DIR>


