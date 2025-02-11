# FakeQuantize in OpenVINO
https://docs.openvino.ai/2025/documentation/openvino-ir-format/operation-sets/operation-specs/quantization/fake-quantize-1.html

definition:
```
    if x <= min(input_low, input_high):
    output = output_low
elif x > max(input_low, input_high):
    output = output_high
else:
    # input_low < x <= input_high
    output = round((x - input_low) / (input_high - input_low) \* (levels-1)) / (levels-1) \* (output_high - output_low) + output_low
```

 - x <= min(input_low, input_high): output = output_low
 - x > max(input_low, input_high):  output = output_high
 - input_low < x <= input_high:

$$
\begin{align}
   q = round(\frac{x - il}{ih - il} * (levels-1)) \\
   output = q * \frac{oh - ol}{levels-1}  + ol
\end{align}
$$

simplified, suppose (ih > il):

$$
\begin{align}
   q = round(\frac{(x - il)}{(ih - il)} * (levels-1)) \\
   q = clamp(q, 0, levels-1) \\
   output = q * \frac{(oh - ol)}{levels-1}  + ol
\end{align}
$$

----------------------------
## Interpretation as Q+DQ
give names to parameters scale(S) & shift(Z)

$$
\begin{align}
   S_i &= \frac{ih - il}{levels-1} \\
   Z_i &= \frac{-il}{S_i}\\
   S_{out} &= \frac{oh - ol}{levels-1} \\
   Z_{out} &= \frac{-ol}{S_o}
\end{align}
$$

using these paramerter, FQ becomes

$$
\begin{align}
   q' &= round(x*\frac{1}{S_i} + Z_i) \tag{a}
\end{align}
$$

$$
\begin{align}
   q_{U} &= clamp(q', 0, levels-1) \tag{b}
\end{align}
$$

$$
\begin{align}
   output &= (q_{U} - Z_{out})* S_o \tag{c}
\end{align}
$$

$q_U$ is unsigned quantized tensor. a small change can make it a signed quantized tensor:

$$
\begin{align}
   Z_0 &= \frac{levels}{2} \\
   q' &= round(x*\frac{1}{S_i} + Z_i - Z_0)   \\
   q_{I} &= clamp(q', -Z_0, Z_0-1) \\
   output &= (q_{I} + Z_0 -  Z_{out})* S_o
\end{align}
$$

here the center value Z0 is substracted before clamp to make a signed quantized value $q_I$ and it was added back later after clamp for mathematical equivalence.

notice:
 - equation (a) is traditional quantization x into q only if Zi is integer:
 - equation (c) is traditional dequantization only if Zo is integer:

thus inputLow/inputHigh/outputLow/outputHigh is gently tuned from statistical result to satisfy these requirements.

# Symetric quantization
In symetric quantize: choose `il` to be `-ih` results in non-integer zero points (since levels is even number)

$$
   Z_i = \frac{-il*(levels-1)}{ih - il} = (levels-1)/2
$$

in symetric quantization, Zi is choosen to be `levels/2`, thus we can increase the range a little by push il to be smaller number

$$
\begin{align}
   (levels-1)/Z_i = -(ih - il)/il = 1 - ih/il \\
   2(1-1/levels) = 1 - ih/il \\
   il = -ih/(1 - 2/levels)
\end{align}
$$

for example:
 - levels=256, U8, Zi=128, il = -1.0078740157480315 * ih

I8 is better choice for symetric quantization beause we can also make zero-point to be 0 if we use I8 for symetric quantization:

$$
   q'_{U8} = clamp(round(x*\frac{1}{S_i} + 128), 0, 255)
$$

$$
   q'_{I8} = clamp(round(x*\frac{1}{S_i}), -128, 127)
$$

# Asymetric quantization

In Asymetric quantization, there is a special case where inputLow=outputLow=0, we can use U8 equation and in this case Zi==Zo=0.

Otherwise, there is no easy way, either `U8` or `I8` requires non-zero zero-points.

# Quantize-only FQ

The actual tensor in memory is stored in quantized form, so FQ is splited as:

 - `Quantize(clamp)` which is fused into `Producer` node as post ops.
 - `Dequantize` is fused into `Consumer` node capable of benefit from quantized representation with additinal zero-point and scales information.

In CPU plugin, most FQ has been split by LPT into `Quantize-only FQ` (with Zo==0 and S_o==1) followed by a Dequantize (further represented as and splitted into a `Subtract` and a `Multiply`)

Many oneDNN primitive has standard support for `Quantize-only FQ` post-ops, which is zero-point & output scale, and this usually is the last post-op before storing to memory as quantized tensor.

To recognize a `Quantize-only FQ` that can be optimized with output-scales post-op, we need to check following two cases:

 - output U8
 - Zi=0 (i.e. inputLow==0)
 - So=1
 - Zo=0

$$
   q'_{U8} = clamp(round(x*\frac{1}{S_i}), 0, 255)
$$

 - output I8
 - Zi=128 (which can be optimized as output I8 with Zi=0)
 - So=1
 - Zout=128 (outputLow = -128)

$$
   q'_{I8} = clamp(round(x*\frac{1}{S_i}), -128, 127)
$$

`Quantize-only FQ` post-ops optimization example:
- `Quantize-only FQ` is the only post-ops of parent node. We optimize FQ by setting the output-scales of parent node. For example, in below pattern, we set
$\frac{1}{S_i}$ as the output scale of `conv` or `inner_produce` to optimize the pattern.
```
      conv --> FQ
      inner_product --> FQ
```
- `Quantize-only FQ` is the last post-ops and `eltwise` post-ops is before FQ. We optimize FQ by setting the output-scales of `eltwise` node. For example, the below pattern, we set $\frac{1}{S_i}$ as the output scale of `eltwise`
```
      conv --> ... --> eltwise --> FQ
      inner_product --> ... --> eltwise --> FQ
```


# Optimization for inference

## push clip after the round step

the original formula of FakeQuantize is changed from:

```
   x = clip(x, cropLow, cropHigh)
   x = x*InputScale + InputShift
   x = round(x)
   x = x*OutputScale + OutputShift
```

into

```
   x = x*InputScale + InputShift
   x = round(x)
   x = clip(x, cropLow2, cropHigh2)
   x = x*OutputScale + OutputShift
```

In practice, the weights of conv/matmul are very likely quantized using per-output-channel setup, making a per-channel dequantization InputScale in following FQ node.  and according to definition of FQ, this also incurred per-channel crop/clip in original inference formula, however, since clip step is actually designed to limit the round result to fit [0,levels-1) range, if we use new formula, the clip step is very likely to become a per-tensor operation again, which can map to high-performance eltwise postOps in oneDNN.

## drop redundant round/clip in some cases

when FQ is the last the fused postOps and output type is s8/u8, oneDNN will saturate the FP32 intermediate result into [-128,127]or[0,255] range and round by default, if OutputScale is 1.0f and OutputShift is integer, we can further bring OutputShift across the clip/round step and fuse it into inputShift:

```
   x = x*InputScale + (InputShift + OutputShift)
   x = round(x)
   x = clip(x, cropLow2+OutputShift, cropHigh2+OutputShift)
```

if we found the clip range `[cropLow2+OutputShift, cropHigh2+OutputShift]` is superset of s8/u8's range, then we know this clip is futile and it can be dropped with the explicit round step.

Note that this is actually a formalization of existing optimization strategy.

## drop round step on some condition

If FQ is not last post Ops, round step can possibly be dropped w/o affect accuracy, considering it only introduces quantization noise (we may need massive test to confirm it so it's not implemented yet).

But existing implementation drops this rounding step inside residual structure (when FQ is followed by a SUM and another FQ), so we still keep this optimization (to avoid performance regression) and formalize it as following:

```
   x = (x*InputScale + InputShift) * OutputScale  + OutputShift
     = x*combinedScale + combinedShift

   x = clip(x, cropLow3, cropHigh3)
```

The combined shift can also be dropped when it's too small comparing to the clip ranges.

## optimize Mappings from fused node to oneDNN's output_scale/postOps:

Existing optimizations mixed the simplification of formula with mapping them into oneDNN's output_scale/postOps, I'm trying to separate these two task by introducing a internal helper class `DnnlPostOpsComposer`, the basic idea is optimized formula decomposed FQ into a serials of basic operation like multiply/add/round/clip(per-tensor or per-OC), but there are no fixed 1v1 mapping between these basic operations and oneDNN attr/postOps, for example, multiply can be mapped to:
  - output_scales for INT8 inference
  - binary for per-OC case
  - eltwise for per-Tensor case

further more, it may even const-folds into previous binary multiply or output scales, for example:

```
  x = (x*A + B)*C = x*(A*C) + (B*C)
```
These optimization should be done in unit of basic operation instead of complex operation like FQ, thus these logic was implemented in class `DnnlPostOpsComposer` and any fused nodes can call its API to take advantage of this common optimization.

So far in non-legacy cases, fused Eltwise & FQ nodes are based this new class.

since there are too many optimizations can be done inside `DnnlPostOpsComposer` and design based on imaginary use case is likely to introduce bugs, so we only add optimization that we've really observed in practical model:

 - use eltwise for all per-tensor operation
 - skip multiply with 1.0f or add with 0.0f
 - first multiply is mapped to output scales
 - Relu is the only preceding postOps when append multiply, we switch the order and fuse the multiply into output scales `relu(x)*s = relu(x*s)`
 - Sum is the only preceding postOps  when append per-tensor multiply, we fuse it into output scale and sum's scale `(x + dst[:])*s = (x*s + s*dst[:])`
 - per-tensor multiply after another eltwise will be fused into that eltwise's scale `eltwise(x, scale, alpha, beta)*s = eltwise(x, (scale*s), alpha, beta)`

## INT8 deconvolution

INT8 `deconvolution_forward` primitive supports many standard stock oneDNN attr&postOps while `convolution_backward` primitive only supports legacy postOps, but we observed that bias is fused as postOps instead of being the bias input of the deconv node, so we :

 - extended `FuseConvolutionMatMulAndBias` as `FuseConvolutionMatMulDeconvAndBias`
 to fuse bias into deconv node
 - extended `Deconv` node to support bias input

which eventually allows bias to be applied with better performance and the per-channel FQ node following deconv to be mapped more efficiently as output scales.
