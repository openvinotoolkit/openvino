# Quantizing with accuracy control {#quantization_w_accuracy_control}

@sphinxdirective

Introduction
####################

This is the advanced quantization flow that allows to apply 8-bit quantization to the model with control of accuracy metric. This is achieved by keeping the most impactful operations within the model in the original precision. The flow is based on the :doc:`Basic 8-bit quantization <basic_qauntization_flow>` and has the following differences:

* Beside the calibration dataset, a **validation dataset** is required to compute accuracy metric. They can refer to the same data in the simplest case.
* **Validation function**, used to compute accuracy metric is required. It can be a function that is already available in the source framework or a custom function.
* Since accuracy validation is run several times during the quantization process, quantization with accuracy control can take more time than the [Basic 8-bit quantization](@ref basic_qauntization_flow) flow.
* The resulted model can provide smaller performance improvement than the :doc:`Basic 8-bit quantization <basic_qauntization_flow>` flow because some of the operations are kept in the original precision.

.. note:: Currently, this flow is available only for models in OpenVINO representation.

The steps for the quantization with accuracy control are described below.

Prepare datasets
####################

This step is similar to the :doc:`Basic 8-bit quantization <basic_qauntization_flow>` flow. The only difference is that two datasets, calibration and validation, are required.

.. tab:: OpenVINO

   .. doxygensnippet:: docs/optimization_guide/nncf/ptq/code/ptq_aa_openvino.py
      :language: python
      :fragment: [dataset]


Prepare validation function
###########################

Validation funtion receives ``openvino.runtime.CompiledModel`` object and validation dataset and returns accuracy metric value. The following code snippet shows an example of validation function for OpenVINO model:

.. tab:: OpenVINO

   .. doxygensnippet:: docs/optimization_guide/nncf/ptq/code/ptq_aa_openvino.py
      :language: python
      :fragment: [validation]


Run quantization with accuracy control

Now, you can run quantization with accuracy control. The following code snippet shows an example of quantization with accuracy control for OpenVINO model:

.. tab:: OpenVINO

   .. doxygensnippet:: docs/optimization_guide/nncf/ptq/code/ptq_aa_openvino.py
      :language: python
      :fragment: [quantization]


``max_drop`` defines the accuracy drop threshold. The quantization process stops when the degradation of accuracy metric on the validation dataset is less than the ``max_drop``.

``nncf.quantize_with_accuracy_control()`` API supports all the parameters of ``nncf.quantize()`` API. For example, you can use ``nncf.quantize_with_accuracy_control()`` to quantize a model with a custom configuration.

See also
####################

* :doc:`Optimizing Models at Training Time <tmo_introduction>`

@endsphinxdirective

