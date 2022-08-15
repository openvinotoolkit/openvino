# Why legacy Post-ops/zero-point in OpenVINO
OpenVINO designed post-ops/zero-points optimization before onednn. Depthwise,FQ postops, and pre-calculated output compensation caused by
input zero-point are introduced into OpenVINO before mature archetecture is developed in onednn. Now, onednn has finalized the post-ops/zero-points
optimization mechanism/API and has some diverges with OV mechanism. To benefit perf improvement from onednn for a long term, no further investment would be applied onto
legacy post-pos/zero-point. So current code would have to support both legacy post-ops/zero-point and onednn post-pos/zero point mechansim.

# Legacy post-ops in OpenVINO
Legacy post-ops in CONV node include depthwise, fake quantization and dwconv.


|legacy post-ops type                   |Fused into conv                           |kernel support list in forked onednn 2.6|
--- | --- | ---|
|depthwise                              |on All ISAs                               |not supported on brgconv_amx and brgconv_avx512_core|
|FQ                                     |on All ISAs                               |not supported on brgconv_amx and brgconv_avx512_core
|dw conv                                |only on avx2 ISA && not support avx512    |only supported on avx2 conv kernel

**dwconv would not fused on CPU above avx2. So no need to consider.**

**depthwise/FQ post-ops may be implemented with onednn binary post-ops in brgconv(brgconv avx512 core and brgconv amx) kernels. Disable brgconv on avx512 core cpu when having depthwise/FQ post-ops** 

# Legacy input zero point in OpenVINO

OpenVINO legacy input zero point can support per-channel input tensor zero point. The pre-calculated output compensation would
be passed into ondnn forked kernel.

|Legacy input zero point                  |Fused into conv                 |kernel support list in forked onednn 2.6
--- | --- | ---|
|**Per-channel input zero point**             |on all platform                   |not supported on jit amx kernel, brgconv_amx and brgconv_avx512_core

Fused legacy input zero point can not supported on amx cpu platforms and will fall back on vnni. The perf would greatly lower than amx.
The stock onednn supports the per-tensor zero points. **Only on AMX platform, per-tensor zero point is fused into conv**

# post-ops attribute in conv

## on avx512-core AMX platform:

|post-ops                            |without zero point          |with per-channel zero point                     |with per-tensor zero point|
--- | --- | ---| ---|
|**without binary**   |attr[0] for all kernels      |attr[0] for legacy zp kernel   |attr[0] for legacy zp,   attr[1] for per-tensor zp|
|**with binary post ops**       |attr[0] for legacy post-ops, attr[1] for brgconv-amx binary post-ops  |attr[0] for legacy zp and legacy post ops  |attr[0] for legacy post-ops + legacy per-channel zp, attr[1] for legacy post ops + per-tensor zp|

**WR attr[1] to use legacy post ops when having per-tensor zero point. Brgconv amx doens't support zero point by now.Switch back to binary postops+per tensor zp when binary perf issue fix in onednn.**

## on AVX512-core with FP32 precision:

**non-AMX kernel will not support per-tensor zero point because of potential conflicts with per-channel zero-point in forked onednn kernel. Only per-channel zero point would be supported.**

|post-ops |without zero point                       |with per-channel/per-tensor zero point|
--- | --- | ---|
|**without binary**     |attr[0] legacy                 |attr[0] for legacy zp kernel +legacy post ops|
|**with binary**        |attr[0] for legacy post-ops    |attr[0] for legacy zp + legacy post ops|


## on AVX512 wth U8 precision:

**non-AMX kernel can't support per-tensor zero point. Only per-channel zero point would be supported.**

|post-ops |without zero point                  |with per-channel zero point
--- | --- | ---|
|**without binary post-ops**     |attr[0] for legacy post-ops       |attr[0] for legacy zp kernel+legacy post ops|
|**with binary post-ops**        |attr[0] for legacy post-ops       |attr[0] for legacy zp + legacy post ops|


# attr[0] and attr[1]
attr[0] is for legacy post-ops or/and legarcy zero point;
attr[1] is to append binary post-ops or append per-tensor zero point on avx512core-amx platform.
**When there is no legagy post ops or legacy zero point, only need attr[0].**