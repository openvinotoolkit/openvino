# Why legacy Post-ops/zero-point in OpenVINO
OpenVINO designed post-ops/zero-points optimization before onednn. Depthwise,FQ postops, and pre-calculated output compensation caused by
input zero-point were introduced into OpenVINO before mature archetecture is developed in onednn. Now, onednn has finalized the post-ops/zero-points
optimization mechanism/API and has some diverges with OV mechanism. To benefit perf improvement from onednn for a long term, no further investment will be applied onto
legacy post-ops/zero-point. So current code will have to support both legacy post-ops/zero-point and onednn stock post-ops/zero point mechanism.

# Legacy post-ops in OpenVINO
Legacy post-ops in CONV node include depthwise, fake quantization and dwconv.


|legacy post-ops type                   |Fused into conv                           |kernel support list in forked onednn 2.6|
--- | --- | ---|
|depthwise                              |on All ISAs                               |not supported on brgconv_amx and brgconv_avx512_core|
|FQ                                     |on All ISAs                               |not supported on brgconv_amx and brgconv_avx512_core
|dw conv                                |only on avx2 ISA && not support avx512    |only supported on avx2 conv kernel

**dwconv is not fused on CPU above avx2. So no need to consider.**

**depthwise/FQ post-ops may be implemented with onednn binary post-ops in brgconv(brgconv avx512 core and brgconv amx) kernels. Due to bad performance of binary post-ops in stock ONEDNN, brgconv is disabled on avx512 core cpu for now when having binary post-ops**

# Legacy per-channel input zero point in OpenVINO

OpenVINO legacy input zero point can support per-channel input tensor zero point. The pre-calculated output compensation will
be passed into onednn forked kernel.

|Legacy input zero point                  |Fused into conv                 |kernel support list in forked onednn 2.6
--- | --- | ---|
|**Per-channel input zero point**             |on all platform                   |not supported on jit amx kernel, brgconv_amx , brgconv_avx512_vnni

Fused legacy per-channel input zero point can not be supported on amx cpu platforms and will fall back on vnni. The perf will be greatly lower than amx.
The stock onednn supports the per-tensor zero points. **Only on AMX/VNNI platform, per-tensor zero point is fused into conv**

# App can enforce BRGCONV kernel
On AMX platform, even if conv fuses binary post-ops, the conv node will try to use brgconv no matter whether app enforces brgconv.

On AVX-512 platform, when conv fuses binary post-ops, the conv node will try to use brgconv only when the app enforce it via rtinfo.
# post-ops attribute in conv

## on avx512-core AMX platform:

|post-ops                            |without zero point          |with per-channel zero point                     |with per-tensor zero point|
--- | --- | ---| ---|
|**without binary**   |attr[0] for all kernels      |attr[0] for legacy zp kernel   |attr[0] for legacy zp,   attr[1] for per-tensor zp|
|**with binary post ops**       |attr[0] for legacy post-ops, attr[1] for brgconv-amx binary post-ops  |attr[0] for legacy zp and legacy post ops  |attr[0] for legacy post-ops + legacy per-channel zp, attr[1] for legacy post ops + per-tensor zp|

**WR attr[1] to use legacy post ops when having per-tensor zero point. Brgconv amx doens't support zero point by now.Switch back to binary postops+per tensor zp when binary perf issue fix in onednn.**

## on AVX512-core non-AMX platform:

**JIT avx512_vnni conv kernel in forked onednn will only support per-channel zero point because of potential conflicts with stock per-tensor zero point. Brgemm avx512_vnni conv kernel
can only support per-tensor stock zero point. OV conv node in CPU plugin needs to support both above 2 zero point mechanisms on avx512-vnni platform.**

|post-ops |without zero point                       |with per-channel zero point       |with per-tensor zero point|
--- | --- | ---| ---|
|**without binary**     |attr[0] for all kernels                 |attr[0] for legacy zp kernel  |attr[0] for legacy zp kernel, attr[1] for per-tensor stock zp|
|**with binary**        |attr[0] for legacy post-ops,attr[1] for enforced brgconv+binary    |attr[0] for legacy zp + legacy post ops    |attr[0] for legacy zp + legacy post ops, attr[1] binary post ops + per-tensor stock zp|


# attr[0] and attr[1]
attr[0] is for legacy post-ops or/and legacy zero point;

attr[1] is aims to append binary post-ops or/and per tensor zero point. 

**When there is no legacy post ops or legacy zero point, only need attr[0].**