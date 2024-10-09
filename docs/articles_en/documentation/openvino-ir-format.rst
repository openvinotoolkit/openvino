OpenVINO IR format
==================


.. meta::
  :description: OpenVINO IR, known as Intermediate Representation, is the result
                of model conversion in OpenVINO and is represented by two files:
                an XML and a binary file.

.. toctree::
   :maxdepth: 1
   :hidden:

   openvino-ir-format/operation-sets
   openvino-ir-format/operation-sets/available-opsets
   openvino-ir-format/operation-sets/operation-specs
   openvino-ir-format/operation-sets/broadcast-rules
   openvino-ir-format/intermediate-representation-int8-inference


OpenVINO IR is the proprietary model format of OpenVINO, benefiting from the full extent
of its features. It is obtained by converting a model from one of the
:doc:`supported formats <../openvino-workflow/model-preparation>`
using the model conversion API or OpenVINO Converter. The process translates common
deep learning operations of the original network to their counterpart representations in
OpenVINO and tunes them with the associated weights and biases.
The resulting OpenVINO IR format contains two files:

* ``.xml`` - Describes the model topology.
* ``.bin`` - Contains the weights and binary data.

:doc:`See why converting to OpenVINO IR is recommended <../openvino-workflow/model-preparation/convert-model-to-ir>`


IR Structure
############

OpenVINO toolkit introduces its own format of graph representation and its own operation set. A graph is represented with two files: an XML file and a binary file. This representation is commonly referred to as the Intermediate Representation or IR.

The XML file describes a model topology using a ``<layer>`` tag for an operation node and an ``<edge>`` tag for a data-flow connection.
Each operation has a fixed number of attributes that define operation flavor used for a node.
For example, the `Convolution` operation has such attributes as ``dilation``, ``stride``, ``pads_begin``, and ``pads_end``.

The XML file does not have big constant values like convolution weights.
Instead, it refers to a part of the accompanying binary file that stores such values in a binary format.

Here is an example of a small IR XML file that corresponds to a graph from the previous section:

.. scrollbox::   

   .. code-block:: cpp

      <?xml version="1.0" ?>
      <net name="model_file_name" version="10">
         <layers>
            <layer id="0" name="input" type="Parameter" version="opset1">
                  <data element_type="f32" shape="1,3,32,100"/> <!-- attributes of operation -->
                  <output>
                     <!-- description of output ports with type of element and tensor dimensions -->
                     <port id="0" precision="FP32">
                        <dim>1</dim>
                        <dim>3</dim>
                        <dim>32</dim>
                        <dim>100</dim>
                     </port>
                  </output>
            </layer>
            <layer id="1" name="conv1/weights" type="Const" version="opset1">
                  <!-- Const is only operation from opset1 that refers to the IR binary file by specifying offset and size in bytes relative to the beginning of the file. -->
                  <data element_type="f32" offset="0" shape="64,3,3,3" size="6912"/>
                  <output>
                     <port id="1" precision="FP32">
                        <dim>64</dim>
                        <dim>3</dim>
                        <dim>3</dim>
                        <dim>3</dim>
                     </port>
                  </output>
            </layer>
            <layer id="2" name="conv1" type="Convolution" version="opset1">
                  <data auto_pad="same_upper" dilations="1,1" output_padding="0,0" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
                  <input>
                     <port id="0">
                        <dim>1</dim>
                        <dim>3</dim>
                        <dim>32</dim>
                        <dim>100</dim>
                     </port>
                     <port id="1">
                        <dim>64</dim>
                        <dim>3</dim>
                        <dim>3</dim>
                        <dim>3</dim>
                     </port>
                  </input>
                  <output>
                     <port id="2" precision="FP32">
                        <dim>1</dim>
                        <dim>64</dim>
                        <dim>32</dim>
                        <dim>100</dim>
                     </port>
                  </output>
            </layer>
            <layer id="3" name="conv1/activation" type="ReLU" version="opset1">
                  <input>
                     <port id="0">
                        <dim>1</dim>
                        <dim>64</dim>
                        <dim>32</dim>
                        <dim>100</dim>
                     </port>
                  </input>
                  <output>
                     <port id="1" precision="FP32">
                        <dim>1</dim>
                        <dim>64</dim>
                        <dim>32</dim>
                        <dim>100</dim>
                     </port>
                  </output>
            </layer>
            <layer id="4" name="output" type="Result" version="opset1">
                  <input>
                     <port id="0">
                        <dim>1</dim>
                        <dim>64</dim>
                        <dim>32</dim>
                        <dim>100</dim>
                     </port>
                  </input>
            </layer>
         </layers>
         <edges>
            <!-- Connections between layer nodes: based on ids for layers and ports used in the descriptions above -->
            <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
            <edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
            <edge from-layer="2" from-port="2" to-layer="3" to-port="0"/>
            <edge from-layer="3" from-port="1" to-layer="4" to-port="0"/>
         </edges>
         <meta_data>
            <!-- This section that is not related to a topology; contains auxiliary information that serves for the debugging purposes. -->
            <MO_version value="2022.3"/>
            <cli_parameters>
                  <blobs_as_inputs value="True"/>
                  <caffe_parser_path value="DIR"/>
                  <data_type value="float"/>

                  ...

                  <!-- Omitted a long list of CLI options that always are put here by MO for debugging purposes. -->

            </cli_parameters>
         </meta_data>
      </net>

The IR does not use explicit data nodes described in the previous section. In contrast, properties of data such as tensor dimensions and their data types are described as properties of input and output ports of operations.

Additional Resources
####################

* :doc:`IR and Operation Sets <openvino-ir-format/operation-sets>`

