# benchdnn

**benchdnn** is a standalone correctness and performance benchmark for
[Intel(R) Math Kernel Library for Deep Neural Networks (Intel(R) MKL-DNN)](/intel/mkl-dnn) library.
The purpose of the benchmark is extended and robust correctness verification of
the primitives provided by MKL-DNN. So far **benchdnn** supports convolutions
and inner products of different data types. It also implicitly tests reorders.


## License
**benchdnn** is licensed under
[Apache License Version 2.0](http://www.apache.org/licenses/LICENSE-2.0).


## Usage (main driver)

**benchdnn** itself is a driver for different implementation specific
harnesses. So far it has harness for Intel MKL-DNN convolution, inner product,
reorder, batch normalization, and harness for testing itself.
The usage:
```
    $ ./benchdnn: [--HARNESS] [--mode=MODE] [-vN|--verbose=N] HARNESS-OPTS
```
where:

 - `HARNESS` is either `conv` [default], `ip`, `shuffle`, `reorder`, `bnorm`, `rnn` or `self`

 - `MODE` -- string that contains flags for benchmark mode. Use `C` or `c` for correctness (used by default), and `P` or `p` for performance

 - `N` -- verbose level (integer from 0 [default] to ...)

 - `HARNESS-OPTS` are passed to the chosen harness

Returns `0` on success (all tests passed), and non-zero in case of any error
happened.


## Usage (convolution harness)

The usage:
```
    [harness-knobs] [conv-desc] ...
```

where *harness-knobs* are:

 - `--cfg={f32, u8s8u8s32, ...}` configuration (see below), default `f32`
 - `--dir={FWD_D (forward data), FWD_B (forward data + bias),FWD_I (forward data inference), BWD_D (backward data), BWD_W (backward weights), BWD_WB (backward weights + bias)}` direction, default `FWD_B`
 - `--alg={DIRECT, WINO}` convolution algorithm, default DIRECT
 - `--merge={NONE, RELU}` merged primitive, default NONE (nothing merged)
 - `--attr="attr_str"` convolution attributes (see in the section below), default `""` (no attributes set)
 - `--mb=N` override minibatch that is specified in convolution description, default `0` (use mb specified in conv desc)
 - `--match=regex` check only convolutions that match with regex, default is `".*"`. Notice: Windows may only interpret string arguments surrounded by double quotation marks.
 - `--skip-impl="str1[:str2]..."` skip implementation (see mkldnn_query_impl_info_str), default `""`
 - `--allow-unimpl=true|false` do not treat unimplemented configuration as an error, default `false`
 - `--perf-template=template-str` set template for performance report (see section *Performance measurements*)
 - `--reset` reset all the parameters set before to default one
 - `-vN|--verbose=N` verbose level, default `0`
 - `--batch=file` use options from the given file (see in subdirectory)

and *conv-desc* is convolution description. The canonical form is:
```
    gXmbXicXihXiwXocXohXowXkhXkwXshXswXphXpwXdhXdwXnS
```
Here X is a number and S is string (n stands for name). Some of the parameters
might be omitted if there is either default one (e.g. if g is not specified
**benchdnn** uses 1) or if the can be computed automatically (e.g. output shape
can be derived from the input one and kernel). Also if either width or height
is not specified than it is assumed height == width. Special symbol `_` is
ignored, hence maybe used as delimiter. See `str2desc()` in conv/conv_aux.cpp
for more details and implicit rules :^)

The attribute string *attr_str* is defined as (new lines for readability):
```
    [irmode={nearest,down};]
    [oscale={none,common,per_oc}[:scale];]
    [post_ops='[{relu,sum[:sum_scale]};]...';]
```

Here `irmode` defines the rounding mode for integer output (default is nearest).

Next, `oscale` stands for output_scales. The first parameter is the policy that
is defined below. The second optional parameter is a scale that specifies
either the one common output scale (for `none` and `common` polices) or a
starting point for `per_oc` policy, which uses many scales. The default scale
is 1.0. Known policies are:

  - `none` (default) means no output scales set (i.e. scale = 1.)
  - `common` corresponds to `mask=0` with common scale factor
  - `per_oc` corresponds to `mask=1<<1` (i.e. output channels) with different scale factors

Next, `post_ops` stands for post operation sequence. Currently supported post
ops are:

  - `relu` with no parameters (i.e. corresponding scale is 1., alg = eltwise_relu, alpha = beta = 0.)
  - `sum` with optional parameter scale (default 1.)

### convolution configurations (aka precision specification)

`--cfg` option specifies what convolution would be used in terms of data type.
Also it defines all the magic with data filling inside. For integer type
saturation is implicitly implied.

Finally configuration defines threshold for computation errors (ideally we
want keep it 0 and it seems to work for now).

The table below shows cases supported by Intel MKL-DNN and corresponding
configurations for **benchdnn**:

|src type | wei type | dst type | acc type | cfg          | notes
|:---     |:---      |:---      |:---      |:---          |:---
| f32     | f32      | f32      | f32      | f32          | inference optimized for sse4.2+, training avx2+
| s16     | s16      | s32      | s32      | s16s16s32s32 | optimized for processors with support of 4vnni, forward pass only (aka FWD_D, FWD_B)
| s32     | s16      | s16      | s32      | s32s16s16s32 | optimized for processors with support of 4vnni, backward wrt data only (aka BWD_D)
| s16     | s32      | s16      | s32      | s16s32s16s32 | optimized for processors with support of 4vnni, backward wrt weights (aka BWD_W, BWD_WB)
| u8      | s8       | f32      | s32      | u8s8f32s32   | optimized for processors with support of avx512vl, forward pass only (aka FWD_D, FWD_B)
| u8      | s8       | s32      | s32      | u8s8s32s32   | same notes as for u8s8f32s32
| u8      | s8       | s8       | s32      | u8s8s8s32    | same notes as for u8s8f32s32
| u8      | s8       | u8       | s32      | u8s8u8s32    | same notes as for u8s8f32s32
| s8      | s8       | f32      | s32      | s8s8f32s32   | same notes as for u8s8f32s32
| s8      | s8       | s32      | s32      | s8s8s32s32   | same notes as for u8s8f32s32
| s8      | s8       | s8       | s32      | s8s8s8s32    | same notes as for u8s8f32s32
| s8      | s8       | u8       | s32      | s8s8u8s32    | same notes as for u8s8f32s32


## Performance measurements

**benchdnn** supports custom performance report. Template is passed via
command line and consists of terminal and nonterminal symbols. Nonterminal
symbols are printed as is. Description of terminal symbols is given below.
There is also a notion of modifiers (marked as @) that change meaning of
terminal symbols, e.g. sign '-' means minimum of (in terms of time). See
table of modifiers below.

> **caution:** threads have to be pinned in order to get consistent frequency

| abbreviation  | description
|:------------  |:-----------
| %d            | problem descriptor
| %D            | expanded problem descriptor (conv parameters in csv format)
| %n            | problem name
| %z            | direction
| %@F           | effective cpu frequency computed as clocks[@] / time[@]
| %O            | number of ops required (padding is not taken into account)
| %@t           | time in ms
| %@c           | time in clocks
| %@p           | ops per second

| modifier  | description
|:--------  |:-----------
|           | default
| -         | min (time) -- default
| 0         | avg (time)
| +         | max (time)
|           |
| K         | Kilo (1e3)
| M         | Mega (1e6)
| G         | Giga (1e9)

The definition of expanded problem descriptor is:
`g,mb,ic,ih,iw,oc,oh,ow,kh,kw,sh,sw,ph,pw`.

The default template can be found in conv/bench_conv.cpp that is defined as
`perf,%n,%d,%GO,%GF,%-t,%-Gp,%0t,%0Gp`. That will produce the following output
in CSV format:
```
string: perf
convolution name
full conv-desc
number of giga ops calculated
effective cpu frequency in GHz (amb clocks[min] / time[min])
minimum time spent in ms
best gigaops (since it corresponds to mimimum time)
average time spent in ms
average gigaops (since it corresponds to average time)
```

## Examples

Run the set of f32 forward convolutions from inputs/conv_all file w/ bias and default minibatch:
```
    $ ./benchdnn --conv \
        --cfg=f32 --dir=FWD_B --batch=inputs/conv_all
```

Run the same but with merged ReLU:
```
    $ ./benchdnn --conv \
        --cfg=f32 --dir=FWD_B --merge=RELU --batch=inputs/conv_all
```

Run the same as previous but also measure performance:
```
    $ ./benchdnn --conv --mode=CORRnPERF \
        --cfg=f32 --dir=FWD_B --merge=RELU --batch=inputs/conv_all
```

> **note**: instead of `CORRnPERF` one can use `CP`, `PC`, `cp`, or `pc`

Run a set of f32 backward convolutions wrt weights with kh=3 and
verbose level set to 2:
```
    $ ./benchdnn --conv -v2 \
        --cfg=f32 --dir=BWD_W --match='.*kh3[^0-9].*' --batch=inputs/conv_all
```

Run a set of u8s8u8s32 backward convolutions wrt data but skip all
the convolutions that will use reference or gemm-based implementation:
```
    $ ./benchdnn --conv \
        --cfg=u8s8u8s32 --dir=BWD_B --skip-impl='ref:gemm' --batch=inputs/conv_all
```

Run explicitly specified 1st forward convolution (including bias) from Alexnet
with the minibatch set to 4, verbose level set to 1 for two given
configurations (`u8s8u8s32` and `f32`):
```
    $ ./benchdnn --conv -v1 \
        --mb=4 --dir=FWD_B \
        --cfg=u8s8u8s32 ic3ih227iw227_oc96oh55ow55_kh11kw11_sh4sw4ph0pw0_n"alexnet:conv1" \
        --cfg=f32 ic3ih227iw227_oc96oh55ow55_kh11kw11_sh4sw4ph0pw0_n"alexnet:conv1"
```

Run batch file for different algorithms (assuming the file only specifies
convolutions and does not include harness options that would override ones
passed in the command line). Also ignore mkldnn_unimplemented errors in case of
Winograd:
```
    $ ./benchdnn --conv \
        --alg=DIRECT --batch=convs.in \
        --allow-unimpl=true \
        --alg=WINO   --batch=convs.in
```

Run a set of u8s8u8s32 forward convolutions w/o bias, skipping
reference implementations and not triggering unimplemented as an error, with
one common output scale set to 0.5 with rounding mode set to down
(via attributes):
```
    $ ./benchdnn --conv \
        --cfg=u8s8u8s32 --dir=FWD_D --skip-impl="ref" --allow-unimpl=true \
        --attr="irmode=down;oscale=common:.5" --batch=inputs/conv_all
```

Almost the same as above (with minor changes), but also add post operation
sequence **(relu, then sum with scale .3, then relu)** using
attributes/mkldnn_post_ops_t:
```
    $ ./benchdnn --conv \
        --cfg=u8s8s32s32 --dir=FWD_B \
        --attr="oscale=common:.5;post_ops='relu;sum:.3;relu'" --batch=inputs/conv_all
```


## Notations / Glossary / Abbreviations

|Abbreviation   | Description
|:---           |:---
| src           | Source image (input image for forward convolution)
| wei           | Weights (aka filter)
| bia           | Bias
| dst           | Destination image (output image for forward convolution)
| acc           | Accumulation (typically in terms of data type)
| ic, oc        | Input/Output channels (aka feature maps)
| ih, iw        | Input height and width
| oh, ow        | Output height and width
| kh, kw        | Kernel (filter, weights) height and width
| sh, sw        | Convolution stride over height and width
| ph, pw        | Convolution top and left padding
| mb            | Minibatch (amount of images processed at once)
| g             | Groups (a way to reduce the amount of computations, see Alexnet topology)
| FWD_{D,B}     | forward w/o and w/ bias
| BWD_{D,W,WB}  | backward wrt data, weights, and weights and bias
| DIRECT, WINO  | convolution algorithm: direct or Winograd based
| NONE, RELU    | merged primitives: nothing or ReLU


## Usage (batch normalization harness)

The usage:
```
    ./benchdnn --bnorm [harness-knobs] bnorm-desc ...
```

where *harness-knobs* are:

 - `--mb=N` override minibatch that is specified in batch normalization description, default `0` (use mb specified in bnorm-desc)
 - `--dir={FWD_D (forward data /training), FWD_I (forward data /inference), BWD_D (backward data), BWD_DW (backward data + weights)}` direction, default `FWD_D`
 - `--dt={f32, s32, ...}` base data type, default `f32`
 - `--fmt={nchw, nChw16c, ...}` data layout, default `nchw`
 - `--flags=[|G|S|R]` batch normalization flags, default `none` (G -- global stats, S -- use scale shift, R -- fuse with ReLU)
 - `--attr="attr_str"` attributes (see in the convolution section above), default `""` (no attributes set)
 - `--match=regex` check only convolutions that match with regex, default is `".*"`. Notice: Windows may only interpret string arguments surrounded by double quotation marks.
 - `--skip-impl="str1[:str2]..."` skip implementation (see mkldnn_query_impl_info_str), default `""`
 - `--perf-template=template-str` set template for performance report (very similar to the convolution one)
 - `--reset` reset all the parameters set before to default one
 - `-vN|--verbose=N` verbose level, default `0`
 - `--batch=file` use options from the given file (see in subdirectory)

and *bnorm-desc* is a batch normalization description. The canonical form is:
```
    mbXicXihXiwXepsYnS
```
Here X is an integer number, Y is a real number, and S is string (n stands for
name). Special symbol `_` is ignored, hence maybe used as delimiter. There are
some implicit rules:
 - if mb is omitted set mb to 2

 - if iw is omitted set iw to ih (and vice versa)

 - if eps is omitted set eps to 1./16


## Installation

**benchdnn** is automatically built with Intel MKL-DNN. For the convenience one
may build **benchdnn** using cmake or make.


## Essence of convolution testing

Intel MKL-DNN supports different data types, such as single precision floating
point (`mkldnn_f32`), signed/unsigned integer of different length
(`mkldnn_{s,u}{8,16,32}`). We need to cover all those cases by tests. It is
essential to test real convolution sizes, since Intel MKL-DNN provides
different optimizations depending on convolution parameters, so there is no
one unified approach inside, which means it would not be enough to test only
few convolutions (aka unit tests).

But even for given convolution the correctness convolution test is not as
simple as it might seem to be at first sight. One of the biggest problem we
encountered is numerical instability. For every output point a lot of
operations may happen. For instance on backward propagation with respect to
filter each filter point requires `mb * oh * ow` operations (see *Notation*
section below). That big amount of compute operations may lead to either
integer overflow or accuracy loss if initial data was chosen inadequately.

These two main things complicate testing. **benchdnn** tries to address these
issues by using integers for initialization with uniform distribution in a
range `[cfg->f_min .. cfg->f_max]`, with the step `cfg->f_step`
(see `struct dt_conf_t` in conv/conv.hpp). `f_min` and `f_max` are chosen so
that most of the result would belong `[cfg->min .. cfg->max]` range. Also
for floating point all integers in both ranges have exact representation (i.e.
the absolute numbers are less than `2^size_of_mantissa`). Uniform distribution
leads to have result uniformly distributed and quite small `f_min/f_max` keep
the result in a reasonable range. Yet another trick: not all the points are
initialized with non-zero values: see `fill_{src,wei,bia,dst}` in
conv/conv.cpp.


## Further plans

Please see TODO.md in **benchdnn** root directory for development plans.


## Issues and contributions

We welcome community contributions to **benchdnn** as well as Intel MKL-DNN.
If you have any ideas or issues please submit an issue or pull request. For
clarity please include ''benchdnn: '' in the title.


## Inspiration

bench{yet another 3 letters where the first one equals second)...
