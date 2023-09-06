.. {#troubleshooting_reshape_errors}

Troubleshooting Reshape Errors
==============================


.. meta::
   :description: In OpenVINO™, you can use several methods to address the issues 
                 of non-reshape-able models and shape collision, which prevent 
                 normal shape propagation.


How To Avoid Shape Collision
############################

Operation semantics may impose restrictions on input shapes of the operation.
Shape collision during shape propagation may be a sign that new shape does not satisfy the restrictions.
Changing the model input shape may result in intermediate operations shape collision. For example, in the following:

* The :doc:`Reshape <openvino_docs_ops_shape_Reshape_1>` operation with a hard-coded output shape value,
* The :doc:`MatMul <openvino_docs_ops_matrix_MatMul_1>` operation with the ``Const`` second input and this input cannot be resized by spatial dimensions due to operation semantics.

Model structure and logic should not change significantly after model reshaping.

* The Global Pooling operation is commonly used to reduce output feature map of classification models output. Having the input of the shape *[N, C, H, W]*, Global Pooling returns the output of the shape *[N, C, 1, 1]*. Model architects usually express Global Pooling with the help of the ``Pooling`` operation with the fixed kernel size *[H, W]*. During spatial reshape, having the input of the shape *[N, C, H1, W1]*, ``Pooling`` with the fixed kernel size *[H, W]* returns the output of the shape *[N, C, H2, W2]*, where *H2* and *W2* are commonly not equal to *1*. It breaks the classification model structure. For example, the public `Inception family models from TensorFlow <https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models>`__ have this issue.

* Changing the model input shape may significantly affect its accuracy. For example, Object Detection models from TensorFlow have resizing restrictions by design. To keep the model valid after the reshape, choose a new input shape that satisfies conditions listed in the ``pipeline.config`` file. For details, refer to the :ref:`Tensorflow Object Detection API models resizing techniques <custom-input-shape>`.

.. _how-to-fix-non-reshape-able-model:

How To Fix Non-Reshape-able Model
#################################

To fix some operators which prevent normal shape propagation:

* see if the issue can be fixed via changing the values of some operators' input. For example, the most common problem of non-reshape-able models is a ``Reshape`` operator with a hard-coded output shape. You can cut-off the hard-coded second input of ``Reshape`` and fill it in with relaxed values. For the following example in the diagram below, the model conversion API command line should read:

  .. code-block:: sh

     mo --input_model path/to/model --input data[8,3,224,224],1:reshaped[2]->[0,-1]`


  With ``1:reshaped[2]``, it is required to cut the second input (counting from zero, so ``1:`` means the second input) of the operation named ``reshaped`` and replace it with a ``Parameter`` with shape ``[2]``.
  With ``->[0 -1]``, this new ``Parameter`` is replaced by a ``Constant`` operator which has the ``[0, -1]`` value.
  Since the ``Reshape`` operator has ``0`` and ``-1`` as specific values, it allows propagating shapes freely without losing the intended meaning of ``Reshape``.   For more information, see :doc:`the specification <openvino_docs_ops_shape_Reshape_1>`.

  .. image:: _static/images/batch_relaxation.png

* transform the model conversion on the back phase. For more information, see the :doc:`How to Convert a Model <openvino_docs_MO_DG_prepare_model_customize_model_optimizer_Customize_Model_Optimizer>`,
* transform OpenVINO Model during the runtime. For more information, see :doc:`OpenVINO Runtime Transformations <openvino_docs_transformations>`,
* modify the original model with the help of the original framework.

