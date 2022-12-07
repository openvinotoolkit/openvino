# Quantizing with accuracy control {#quantization_w_accuracy_control}

## Introduction

This is the advanced quantization flow that allows to apply 8-bit quantization to the model with control of accuracy metric. This is achieved by keeping the most impactful operations within the model in the original precision. The flow is based on the [Basic 8-bit quantization](@ref basic_qauntization_flow) and has the following differences:
* Besided the calibration dataset, a **validation dataset** is required to compute accuracy metric. They can refer to the same data in the simplest case.
* **Validation function**, used to compute accuracy metric is required. It can be a function that is already available in the source framework or a custom function.
* Since accuracy validation is run several times during the quantization process, quantization with accuracy control can take more time than the [Basic 8-bit quantization](@ref basic_qauntization_flow) flow.
* The resulted model can provide smaller performance improvement than the [Basic 8-bit quantization](@ref basic_qauntization_flow) flow bacause some of the operations are kept in the original precision.

The steps for the quantizatation with accuracy control are described below.

## Prepare datasets

This step is similar to the [Basic 8-bit quantization](@ref basic_qauntization_flow) flow. The only difference is that two datasets, calibration and validation, are required.

@sphinxtabset

@sphinxtab{PyTorch}

@snippet docs/optimization_guide/nncf/ptq/code/ptq_aa_openvino.py dataset

@endsphinxtab

@endsphinxtabset



