.. {#openvino_docs_OV_UG_ways_to_get_stateful_model}

Obtaining a Stateful OpenVINO Model
====================================

If the original framework does not offer a dedicated API for working with states, the
resulting OpenVINO IR model will not be stateful by default. This means it will not contain
either a state or the :doc:`Assign <openvino_docs_ops_infrastructure_Assign_6>` and
:doc:`ReadValue <openvino_docs_ops_infrastructure_ReadValue_6>` operations. You can still
make such models stateful (:doc:`see benefits <openvino_docs_OV_UG_stateful_models_intro>`),
and you have three ways to do it:

* `Optimum-Intel <https://github.com/huggingface/optimum-intel>`__ - an automated solution
  applicable to a selection of models (not covered by this article, for a usage guide
  refer to the :doc:`Optimize and Deploy Generative AI Models <gen_ai_guide>` article).
* :ref:`MakeStateful transformation <ov_ug_make_stateful>` - to choose which pairs of
  Parameter and Result to replace.
* :ref:`LowLatency2 transformation <ov_ug_low_latency>` - to detect and replace Parameter
  and Result pairs connected to hidden and cell state inputs of LSTM/RNN/GRU operations
  or Loop/TensorIterator operations.


.. _ov_ug_make_stateful:

MakeStateful Transformation
###########################

The MakeStateful transformation changes the structure of the model by replacing the
user-defined pairs of Parameter and Results with the Assign and ReadValue operations:

.. image:: _static/images/make_stateful_simple.svg
   :alt: diagram of MakeStateful Transformation
   :scale: 90 %
   :align: center

**Only strict syntax is supported**. As shown in the example below, the transformation call
must be enclosed in double quotes "MakeStateful[...]", tensor names - in single quotes
without spaces 'tensor_name_1'.

**State naming rule**: in most cases, the name of a state is a concatenation of the
Parameter/Result tensor names. If there are no tensor names,
:doc:`friendly names <openvino_docs_transformations>` are used.


**Examples:**

.. image:: _static/images/make_stateful_detailed.png
   :alt: detailed diagram of MakeStateful Transformation
   :align: center


.. tab-set::

   .. tab-item:: C++

      .. tab-set::

         .. tab-item:: Using tensor names

            .. doxygensnippet:: docs/snippets/ov_stateful_models_intro.cpp
               :language: cpp
               :fragment: [ov:make_stateful_tensor_names]

         .. tab-item:: Using Parameter/Result operations

            .. doxygensnippet:: docs/snippets/ov_stateful_models_intro.cpp
               :language: cpp
               :fragment: [ov:make_stateful_ov_nodes]

   .. tab-item:: command line

      .. tab-set::

         .. tab-item:: Using tensor names

            .. code-block:: sh

               --input_model <INPUT_MODEL> --transform "MakeStateful[param_res_names={'tensor_name_1':'tensor_name_4','tensor_name_3':'tensor_name_6'}]"




.. _ov_ug_low_latency:

LowLatency2 Transformation
##########################

The LowLatency2 transformation changes the structure of a model containing
:doc:`TensorIterator <openvino_docs_ops_infrastructure_TensorIterator_1>`
and :doc:`Loop <openvino_docs_ops_infrastructure_Loop_5>` by automatically detecting
and replacing pairs of Parameter and Results with the Assign and ReadValue operations,
as illustrated by the following example:

.. image:: _static/images/applying_low_latency_2.svg
   :alt: diagram of LowLatency Transformation
   :align: center

After applying the transformation, ReadValue operations can receive other operations as
input, as shown in the picture above. These inputs should set the initial value for the
initialization of ReadValue operations. However, such initialization is not supported in
the current State API implementation. Input values are ignored, and the initial values
for the ReadValue operations are set to zeros unless the user specifies otherwise via
:ref:`State API <ov_ug_state_api>`.

Applying LowLatency2 Transformation
++++++++++++++++++++++++++++++++++++

1. Get :doc:`ov::Model <openvino_docs_OV_UG_Model_Representation>`, for example:

   .. tab-set::

      .. tab-item:: C++

         .. doxygensnippet:: docs/snippets/ov_stateful_models_intro.cpp
            :language: cpp
            :fragment: [ov:get_ov_model]

2. Change the number of iterations inside TensorIterator/Loop nodes in the model using the
   :doc:`Reshape <openvino_docs_OV_UG_ShapeInference>` feature.

   For example, the *sequence_lengths* dimension of the model input > 1, it means the
   TensorIterator layer has the number_of_iterations > 1. You can reshape the model
   inputs to set the *sequence_dimension* to exactly 1.

   .. tab-set::

      .. tab-item:: C++

         .. doxygensnippet:: docs/snippets/ov_stateful_models_intro.cpp
            :language: cpp
            :fragment: [ov:reshape_ov_model]

   **Unrolling**: If the LowLatency2 transformation is applied to a model containing
   TensorIterator/Loop nodes with exactly one iteration inside, these nodes are unrolled.
   Otherwise, the nodes remain as they are. See the picture above for more details.

3. Apply LowLatency2 transformation.

   .. tab-set::

      .. tab-item:: C++

         .. doxygensnippet:: docs/snippets/ov_stateful_models_intro.cpp
            :language: cpp
            :fragment: [ov:apply_low_latency_2]


   (Optional) Use Const Initializer argument:

   By default, the LowLatency2 transformation inserts a constant subgraph of the same shape
   as the previous input node. The initializing value for ReadValue nodes is set to zero.
   For more information, see the picture below. You can disable the insertion of this subgraph
   by setting the ``use_const_initializer`` argument to ``false``.

   .. tab-set::

      .. tab-item:: C++

         .. doxygensnippet:: docs/snippets/ov_stateful_models_intro.cpp
            :language: cpp
            :fragment: [ov:low_latency_2_use_parameters]


   .. image:: _static/images/llt2_use_const_initializer.svg
      :alt: diagram of constant subgraph initialization
      :align: center

   **State naming rule:**  the name of a state is a concatenation of several names: the original
   TensorIterator operation, the parameter of the body, and an additional suffix "variable_" + id
   (zero-based indexing, new indexing for each TensorIterator). You can use these rules to predict
   the name of the inserted state after applying the transformation. For example:

   .. tab-set::

      .. tab-item:: C++

         .. doxygensnippet:: docs/snippets/ov_stateful_models_intro.cpp
            :language: cpp
            :fragment: [ov:low_latency_2]


4. Use state API. See sections :ref:`OpenVINO State API <ov_ug_state_api>`,
   :ref:`Stateful Model Inference <ov_ug_stateful_model_inference>`.

   .. image:: _static/images/low_latency_limitation_2.svg
      :alt: diagram showing low latency limitation
      :scale: 70 %
      :align: center

   The only way to change the number iterations of TensorIterator/Loop layer is to use the
   :doc:`Reshape <openvino_docs_OV_UG_ShapeInference>` feature. However, some models may be
   non-reshapable, typically because the value of shapes is hardcoded in a constant
   somewhere in the model.

   In such a case, trim non-reshapable layers via
   :doc:`Model Optimizer command-line <openvino_docs_MO_DG_prepare_model_convert_model_Converting_Model>`
   arguments: ``--input`` and ``--output``.

   For example, the parameter and the problematic constant in the picture above can be
   trimmed using the ``--input Reshape_layer_name`` command-line option. The problematic
   constant can be also replaced using OpenVINO, as shown in the following example:

   .. tab-set::

      .. tab-item:: C++

         .. doxygensnippet:: docs/snippets/ov_stateful_models_intro.cpp
            :language: cpp
            :fragment: [ov:replace_const]



Obtaining TensorIterator/Loop Operations using Model Optimizer
###############################################################

**ONNX and frameworks supported via ONNX format:** *LSTM, RNN, GRU* original layers are
converted to the GRU/RNN/LSTM Sequence operations. *ONNX Loop* layer is converted to the
OpenVINO Loop operation.

**TensorFlow:** *BlockLSTM* is converted to a TensorIterator operation. TensorIterator
body contains LSTM Cell operation. Modifications such as Peepholes and InputForget are
not supported. The *While* layer is converted to a TensorIterator. TensorIterator body
can contain any supported operations. However, dynamic cases where the count of iterations
cannot be calculated during shape inference (Model Optimizer conversion) are not supported.

**TensorFlow2:** *While* layer is converted to a Loop operation. The Loop body can contain
any supported operations.



Creating a Model via OpenVINO API
##################################

The main approach to obtaining stateful OpenVINO IR models is converting from other
frameworks. Nonetheless, it is possible to create a model from scratch. Check how to
do so in the :doc:`Build OpenVINO Model section <openvino_docs_OV_UG_Model_Representation>`.

Here is also an example of how ``ov::SinkVector`` is used to create ``ov::Model``. For a
model with states, except inputs and outputs, ``Assign`` nodes should also point to ``Model``
to avoid deleting it during graph transformations. You can do it with the constructor, as in
the example, or with the `add_sinks(const SinkVector& sinks)` method. Also, you can delete
a sink from `ov::Model` after deleting the node from the graph with the `delete_sink()` method.

.. tab-set::

   .. tab-item:: C++

      .. doxygensnippet:: docs/snippets/ov_stateful_models_intro.cpp
         :language: cpp
         :fragment: [ov:state_network]

