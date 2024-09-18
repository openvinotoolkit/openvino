Obtaining a Stateful OpenVINO Model
======================================

If the original framework does not offer a dedicated API for working with states, the
resulting OpenVINO IR model will not be stateful by default. This means it will not contain
either a state or the :doc:`Assign <../../../documentation/openvino-ir-format/operation-sets/operation-specs/infrastructure/assign-6>` and
:doc:`ReadValue <../../../documentation/openvino-ir-format/operation-sets/operation-specs/infrastructure/read-value-6>` operations. You can still
make such models stateful (:doc:`see benefits <../stateful-models>`),
and you have three ways to do it:

* `Optimum-Intel <https://github.com/huggingface/optimum-intel>`__ - an automated solution
  applicable to a selection of models (not covered by this article, for a usage guide
  refer to the :doc:`LLM Inference with Hugging Face and Optimum Intel <../../../learn-openvino/llm_inference_guide>` article).
* :ref:`MakeStateful transformation <ov_ug_make_stateful>` - to choose which pairs of
  Parameter and Result to replace.
* :ref:`LowLatency2 transformation <ov_ug_low_latency>` - to detect and replace Parameter
  and Result pairs connected to hidden and cell state inputs of LSTM/RNN/GRU operations
  or Loop/TensorIterator operations.


.. _ov_ug_make_stateful:

MakeStateful Transformation
###############################

The MakeStateful transformation changes the structure of the model by replacing the
user-defined pairs of Parameter and Results with the Assign and ReadValue operations:

.. image:: ../../../assets/images/make_stateful_simple.svg
   :alt: diagram of MakeStateful Transformation
   :scale: 90 %
   :align: center

**Only strict syntax is supported**. As shown in the example below, the transformation call
must be enclosed in double quotes "MakeStateful[...]", tensor names - in single quotes
without spaces 'tensor_name_1'.

**State naming rule**: in most cases, the name of a state is a concatenation of the
Parameter/Result tensor names. If there are no tensor names,
:doc:`friendly names <../../../documentation/openvino-extensibility/transformation-api>` are used.


**Examples:**

.. image:: ../../../assets/images/make_stateful_detailed.png
   :alt: detailed diagram of MakeStateful Transformation
   :align: center


.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. tab-set::

         .. tab-item:: Using tensor names
            :sync: using-tensor-names

            .. doxygensnippet:: docs/articles_en/assets/snippets/ov_stateful_models_intro.py
               :language: py
               :fragment: [ov:make_stateful_tensor_names]

         .. tab-item:: Using Parameter/Result operations
            :sync: using-ops

            .. doxygensnippet:: docs/articles_en/assets/snippets/ov_stateful_models_intro.py
               :language: py
               :fragment: [ov:make_stateful_ov_nodes]

   .. tab-item:: C++
      :sync: cpp

      .. tab-set::

         .. tab-item:: Using tensor names
            :sync: using-tensor-names

            .. doxygensnippet:: docs/articles_en/assets/snippets/ov_stateful_models_intro.cpp
               :language: cpp
               :fragment: [ov:make_stateful_tensor_names]

         .. tab-item:: Using Parameter/Result operations
            :sync: using-ops

            .. doxygensnippet:: docs/articles_en/assets/snippets/ov_stateful_models_intro.cpp
               :language: cpp
               :fragment: [ov:make_stateful_ov_nodes]

   .. tab-item:: command line
      :sync: command-line

      .. tab-set::

         .. tab-item:: Using tensor names
            :sync: using-tensor-names

            .. code-block:: sh

               --input_model <INPUT_MODEL> --transform "MakeStateful[param_res_names={'tensor_name_1':'tensor_name_4','tensor_name_3':'tensor_name_6'}]"


.. _ov_ug_low_latency:

LowLatency2 Transformation
###############################

The LowLatency2 transformation changes the structure of a model containing
:doc:`TensorIterator <../../../documentation/openvino-ir-format/operation-sets/operation-specs/infrastructure/tensor-iterator-1>`
and :doc:`Loop <../../../documentation/openvino-ir-format/operation-sets/operation-specs/infrastructure/loop-5>` by automatically detecting
and replacing pairs of Parameter and Results with the Assign and ReadValue operations,
as illustrated by the following example:

.. image:: ../../../assets/images/applying_low_latency_2.svg
   :alt: diagram of LowLatency Transformation
   :align: center

After applying the transformation, ReadValue operations can receive other operations as
input, as shown in the picture above. These inputs should set the initial value for the
initialization of ReadValue operations. However, such initialization is not supported in
the current State API implementation. Input values are ignored, and the initial values
for the ReadValue operations are set to zeros unless the user specifies otherwise via
:doc:`State API <../stateful-models>`.

To apply LowLatency2 Transformation, follow the instruction below:

1. Get :doc:`ov::Model <../integrate-openvino-with-your-application/model-representation>`,
   for example:

   .. tab-set::

      .. tab-item:: Python
         :sync: py

         .. doxygensnippet:: docs/articles_en/assets/snippets/ov_stateful_models_intro.py
            :language: py
            :fragment: [ov:get_ov_model]

      .. tab-item:: C++
         :sync: cpp

         .. doxygensnippet:: docs/articles_en/assets/snippets/ov_stateful_models_intro.cpp
            :language: cpp
            :fragment: [ov:get_ov_model]


2. Change the number of iterations inside TensorIterator/Loop nodes in the model using the
   :doc:`Reshape <../changing-input-shape>` feature.

   For example, the *sequence_lengths* dimension of the model input > 1, it means the
   TensorIterator layer has the number_of_iterations > 1. You can reshape the model
   inputs to set the *sequence_dimension* to exactly 1.

   .. tab-set::

      .. tab-item:: Python
         :sync: py

         .. doxygensnippet:: docs/articles_en/assets/snippets/ov_stateful_models_intro.py
            :language: py
            :fragment: [ov:reshape_ov_model]

      .. tab-item:: C++
         :sync: cpp

         .. doxygensnippet:: docs/articles_en/assets/snippets/ov_stateful_models_intro.cpp
            :language: cpp
            :fragment: [ov:reshape_ov_model]


   **Unrolling**: If the LowLatency2 transformation is applied to a model containing
   TensorIterator/Loop nodes with exactly one iteration inside, these nodes are unrolled.
   Otherwise, the nodes remain as they are. See the picture above for more details.

3. Apply LowLatency2 transformation.

   .. tab-set::

      .. tab-item:: Python
         :sync: py

         .. doxygensnippet:: docs/articles_en/assets/snippets/ov_stateful_models_intro.py
            :language: py
            :fragment: [ov:apply_low_latency_2]

      .. tab-item:: C++
         :sync: cpp

         .. doxygensnippet:: docs/articles_en/assets/snippets/ov_stateful_models_intro.cpp
            :language: cpp
            :fragment: [ov:apply_low_latency_2]


   (Optional) Use Const Initializer argument:

   By default, the LowLatency2 transformation inserts a constant subgraph of the same shape
   as the previous input node. The initializing value for ReadValue nodes is set to zero.
   For more information, see the picture below. You can disable the insertion of this subgraph
   by setting the ``use_const_initializer`` argument to ``false``.

   .. tab-set::

      .. tab-item:: Python
         :sync: py

         .. doxygensnippet:: docs/articles_en/assets/snippets/ov_stateful_models_intro.py
            :language: py
            :fragment: [ov:low_latency_2_use_parameters]

      .. tab-item:: C++
         :sync: cpp

         .. doxygensnippet:: docs/articles_en/assets/snippets/ov_stateful_models_intro.cpp
            :language: cpp
            :fragment: [ov:low_latency_2_use_parameters]


   .. image:: ../../../assets/images/llt2_use_const_initializer.svg
      :alt: diagram of constant subgraph initialization
      :align: center

   **State naming rule:**  the name of a state is a concatenation of several names: the
   original TensorIterator operation, the parameter of the body, and an additional suffix
   ``"variable_"`` + id (zero-based indexing, new indexing for each TensorIterator). You can
   use these rules to predict the name of the inserted state after applying the transformation.
   For example:

   .. tab-set::

      .. tab-item:: Python
         :sync: py

         .. doxygensnippet:: docs/articles_en/assets/snippets/ov_stateful_models_intro.py
            :language: py
            :fragment: [ov:low_latency_2]

      .. tab-item:: C++
         :sync: cpp

         .. doxygensnippet:: docs/articles_en/assets/snippets/ov_stateful_models_intro.cpp
            :language: cpp
            :fragment: [ov:low_latency_2]


4. Use state API. See sections :doc:`OpenVINO State API <../stateful-models>`,
   :ref:`Stateful Model Inference <ov_ug_stateful_model_inference>`.

   .. image:: ../../../assets/images/low_latency_limitation_2.svg
      :alt: diagram showing low latency limitation
      :scale: 70 %
      :align: center

   The only way to change the number iterations of TensorIterator/Loop layer is to use the
   :doc:`Reshape <../changing-input-shape>` feature. However, some models may be
   non-reshapable, typically because the value of shapes is hardcoded in a constant
   somewhere in the model.

   In such a case, trim non-reshapable layers via
   :doc:`Conversion Parameters <../../model-preparation/conversion-parameters>`:
   ``--input`` and ``--output``. For example, check the `OpenVINO Model Conversion Tutorial <https://docs.openvino.ai/2024/notebooks/convert-to-openvino-with-output.html>`__.

   As for the parameter and the problematic constant in the picture above, it can be
   trimmed by using the ``--input Reshape_layer_name`` command-line option. The problematic
   constant can be also replaced using OpenVINO, as shown in the following example:

   .. tab-set::

      .. tab-item:: Python
         :sync: py

         .. doxygensnippet:: docs/articles_en/assets/snippets/ov_stateful_models_intro.py
            :language: py
            :fragment: [ov:replace_const]

      .. tab-item:: C++
         :sync: cpp

         .. doxygensnippet:: docs/articles_en/assets/snippets/ov_stateful_models_intro.cpp
            :language: cpp
            :fragment: [ov:replace_const]


Stateful Model from Scratch
##################################

The main approach to obtaining stateful OpenVINO IR models is converting from other
frameworks. Nonetheless, it is possible to create a model from scratch. Check how to
do so in the :doc:`Build OpenVINO Model section <../integrate-openvino-with-your-application/model-representation>`.

Here is also an example of how ``ov::SinkVector`` is used to create ``ov::Model``. For a
model with states, except inputs and outputs, ``Assign`` nodes should also point to ``Model``
to avoid deleting it during graph transformations. You can do it with the constructor, as in
the example, or with the `add_sinks(const SinkVector& sinks)` method. Also, you can delete
a sink from `ov::Model` after deleting the node from the graph with the `delete_sink()` method.

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_stateful_models_intro.py
         :language: py
         :fragment: [ov:stateful_model]

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/articles_en/assets/snippets/ov_stateful_models_intro.cpp
         :language: cpp
         :fragment: [ov:stateful_model]


.. note::

   **ONNX and frameworks supported via ONNX format:** *LSTM, RNN, GRU* original layers are
   converted to the GRU/RNN/LSTM Sequence operations. *ONNX Loop* layer is converted to the
   OpenVINO Loop operation.

   **TensorFlow:** *BlockLSTM* is converted to a TensorIterator operation. The TensorIterator
   body contains LSTM Cell operation. Modifications such as Peepholes and InputForget are
   not supported. The *While* layer is converted to a TensorIterator. The TensorIterator body
   can contain any supported operations. However, dynamic cases where the count of iterations
   cannot be calculated during shape inference are not supported.

   **TensorFlow2:** *While* layer is converted to a Loop operation. The Loop body can contain
   any supported operations.
