# FakeQuantize in OpenVINO
https://docs.openvino.ai/latest/openvino_docs_ops_quantization_FakeQuantize_1.html

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

# FQCommon

The actual tensor is stored in memory as floating point type. So `round` is not needed in this case. The output can be simplified as:

$$
\begin{align}
   y =(x-il)*\frac{oh-ol}{ih-il} + ol \\
   y =x*\frac{oh-ol}{ih-il} + c \\
   c = -il*\frac{oh-ol}{ih-il} + ol
\end{align}
$$

 If the following conditions are ture, FQ can be optimized with output-scales $\frac{oh-ol}{ih-il}$.

 $$
 |c/(oh-ol)| = |\frac{ol}{oh-ol} -\frac{il}{ih-il}| < 0.01
 $$