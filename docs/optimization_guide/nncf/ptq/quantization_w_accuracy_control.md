# Quantizing with accuracy control {#quantization_w_accuracy_control}

## Introduction

This is the advanced quantization flow that allows to apply 8-bit quantization to the model with control of accuracy metric. This is achieved by keeping the most impactful operations within the model in the original precision. The flow is based on the [Basic 8-bit quantization](@ref basic_qauntization_flow) and has the following differences:
* Besided the calibration dataset, a **validation dataset** is required to compute accuracy metric. They can refer to the same data in the simplest case.
* **Validation function**, used to compute accuracy metric is required. It can be a function that is already available in the source framework or a custom function.
* Since accuracy validation is run several times during the quantization process, quantization with accuracy control can take more time than the [Basic 8-bit quantization](@ref basic_qauntization_flow) flow.
* The resulted model can provide smaller performance improvement than the [Basic 8-bit quantization](@ref basic_qauntization_flow) flow because some of the operations are kept in the original precision.

> **NOTE**: Currently, this flow is available only for models in OpenVINO representation.

The steps for the quantizatation with accuracy control are described below.

## Prepare datasets

This step is similar to the [Basic 8-bit quantization](@ref basic_qauntization_flow) flow. The only difference is that two datasets, calibration and validation, are required.

@sphinxtabset

@sphinxtab{OpenVINO}

@snippet docs/optimization_guide/nncf/ptq/code/ptq_aa_openvino.py dataset

@endsphinxtab

@endsphinxtabset

## Prepare validation function

Validation funtion receives `openvino.runtime.CompiledModel` object and 
validation dataset and returns accuracy metric value. The following code snippet shows an example of validation function for OpenVINO model:

@sphinxtabset

@sphinxtab{OpenVINO}

@snippet docs/optimization_guide/nncf/ptq/code/ptq_aa_openvino.py validation

@endsphinxtab

@endsphinxtabset

## Run quantization with accuracy control

Now, you can run quantization with accuracy control. The following code snippet shows an example of quantization with accuracy control for OpenVINO model:  

@sphinxtabset

@sphinxtab{OpenVINO}

@snippet docs/optimization_guide/nncf/ptq/code/ptq_aa_openvino.py quantization

@endsphinxtab

@endsphinxtabset

`max_drop` defines the accuracy drop threshold. The quantization process stops when the degradation of accuracy metric on the validation dataset is less than the `max_drop`. 

`nncf.quantize_with_accuracy_control()` API supports all the parameters of `nncf.quantize()` API. For example, you can use `nncf.quantize_with_accuracy_control()` to quantize a model with a custom configuration.

## See also

* [Optimizing Models at Training Time](@ref tmo_introduction)


