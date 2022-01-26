# Quantization {#pot_compression_algorithms_quantization_README}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:
   
   DefaultQuantization Algorithm <pot_compression_algorithms_quantization_default_README>
   AccuracyAwareQuantization Algorithm <pot_compression_algorithms_quantization_accuracy_aware_README>
   TunableQuantization Algorithm <pot_compression_algorithms_quantization_tunable_quantization_README>

@endsphinxdirective

The primary optimization feature of the Post-training Optimization Tool (POT) is uniform quantization. In general,
this method supports an arbitrary number of bits, greater or equal to two, which represents weights and activations.
During the quantization process, the method inserts [FakeQuantize](@ref openvino_docs_ops_quantization_FakeQuantize_1)
operations into the model graph automatically based on a predefined hardware target in order to produce the most
hardware-friendly optimized model:
![](../../../../../docs/images/convolution_quantization.png)

After that, different quantization algorithms can tune the `FakeQuantize` parameters or remove some of them in order to
meet the accuracy criteria. The resulting *fakequantized* models are interpreted and transformed to real low-precision
models during inference at the OpenVINO™ Inference Engine runtime giving real performance improvement.

## Quantization Algorithms

Currently, the POT provides two algorithms for 8-bit quantization, which are verified and provide stable results on a
wide range of DNN models:
*  **DefaultQuantization** is a default method that provides fast and in most cases accurate results for 8-bit
   quantization. For details, see the [DefaultQuantization Algorithm](@ref pot_compression_algorithms_quantization_default_README) documentation.

*  **AccuracyAwareQuantization** enables remaining at a predefined range of accuracy drop after quantization at the cost
   of performance improvement. It may require more time for quantization. For details, see the
   [AccuracyAwareQuantization Algorithm](@ref pot_compression_algorithms_quantization_accuracy_aware_README) documentation.

## Quantization Formula

Quantization is parametrized by clamping the range and the number of quantization levels:

\f[  
output = \frac{\left\lfloor (clamp(input; input\_low, input\_high)-input\_low)  *s\right \rceil}{s} + input\_low\\  
\f]

\f[
clamp(input; input\_low, input\_high) = min(max(input, input\_low), input\_high)))
\f]

\f[
s=\frac{levels-1}{input\_high - input\_low}
\f]

In the formulas:
* `input_low` and `input_high` represent the quantization range 
* \f[\left\lfloor\cdot\right \rceil\f] denotes rounding to the nearest integer

The POT supports symmetric and asymmetric quantization of weights and activations, which are controlled by the `preset`.
The main difference between them is that in the symmetric mode the floating-point zero is mapped directly to the integer
zero, while in asymmetric the mode it can be an arbitrary integer number. In any mode, the floating-point zero is mapped
directly to the quant without rounding an error. See this [tutorial](@ref pot_docs_BestPractices) for details.

Below is the detailed description of quantization formulas for both modes. These formulas are used both in the POT to
quantize weights of the model and in the OpenVINO™ Inference Engine runtime when quantizing activations during the
inference.

####  Symmetric Quantization

The formula is parametrized by the `scale` parameter that is tuned during the quantization process:

\f[
input\_low=scale*\frac{level\_low}{level\_high}
\f]

\f[
input\_high=scale
\f]


Where `level_low` and `level_high` represent the range of the discrete signal.
* For weights:

\f[
level\_low=-2^{bits-1}+1
\f]

\f[
level\_high=2^{bits-1}-1
\f]

\f[
levels=255
\f]

* For unsigned activations:

\f[
level\_low=0
\f]

\f[
level\_high=2^{bits}-1
\f]

\f[
levels=256
\f]

* For signed activations:

\f[
level\_low=-2^{bits-1}
\f]

\f[
level\_high=2^{bits-1}-1
\f]


\f[
levels=256
\f]

####  Asymmetric Quantization

The quantization formula is parametrized by `input_low` and `input_range` that are tunable parameters:

\f[
input\_high=input\_low + input\_range
\f]

\f[
levels=256
\f]

For weights and activations the following quantization mode is applied:

\f[
{input\_low}' = min(input\_low, 0)
\f]

\f[
{input\_high}' = max(input\_high, 0)
\f]

\f[
ZP= \left\lfloor \frac{-{input\_low}'*(levels-1)}{{input\_high}'-{input\_low}'} \right \rceil 
\f]

\f[
{input\_high}''=\frac{ZP-levels+1}{ZP}*{input\_low}'
\f]

\f[
{input\_low}''=\frac{ZP}{ZP-levels+1}*{input\_high}'
\f]

\f[
{input\_low,input\_high} = \begin{cases} {input\_low}',{input\_high}', & ZP \in $\{0,levels-1\}$ \\ {input\_low}',{input\_high}'', & {input\_high}'' - {input\_low}' > {input\_high}' - {input\_low}'' \\ {input\_low}'',{input\_high}', & {input\_high}'' - {input\_low}' <= {input\_high}' - {input\_low}''\\ \end{cases}
\f]
