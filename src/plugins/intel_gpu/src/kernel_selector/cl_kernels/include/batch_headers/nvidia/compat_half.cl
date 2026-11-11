// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Copyright (C) 2026 FUJITSU LIMITED
//
// half support for the native NVIDIA OpenCL compiler, which enables cl_khr_fp16 but leaves most of the half built-ins
// out: the HALF_* limits, math overloads, convert_* families, as_* reinterprets, vloadN / vstoreN, the relational
// built-ins and the half atomics of cl_ext_float_atomics that istft_ref.cl uses. The convert_* section also carries the
// emulation of Intel's out-of-range float to integer conversion, which is not specific to half but belongs to the same
// family of overrides.
//
// kernels_cache.cpp emits this file as part of the NVIDIA prelude; nvidia_prelude_names there fixes the order.
// Dependencies are listed here instead of as #include lines: primitive_db_gen.py strips #include from batch headers
// anyway, and its dependency sort fails on include graphs that reach the same header along two paths.
//
// Depends on:
//     nvidia/compat_common.cl   cl_khr_fp16

#ifdef __NV_CL_C_VERSION

// --------------------------------------------------------------------------------------------------------------------
// Limits
// --------------------------------------------------------------------------------------------------------------------

#ifndef HALF_MAX
#define HALF_MAX 0x1.ffcp15h
#endif
#ifndef HALF_MIN
#define HALF_MIN 0x1.0p-14h
#endif
#ifndef HALF_EPSILON
#define HALF_EPSILON 0x1.0p-10h
#endif
#ifndef HALF_DIG
#define HALF_DIG 3
#endif
#ifndef HALF_MANT_DIG
#define HALF_MANT_DIG 11
#endif
#ifndef HALF_MAX_10_EXP
#define HALF_MAX_10_EXP +4
#endif
#ifndef HALF_MAX_EXP
#define HALF_MAX_EXP +16
#endif
#ifndef HALF_MIN_10_EXP
#define HALF_MIN_10_EXP -4
#endif
#ifndef HALF_MIN_EXP
#define HALF_MIN_EXP -13
#endif
#ifndef HALF_RADIX
#define HALF_RADIX 2
#endif

// --------------------------------------------------------------------------------------------------------------------
// half <-> float conversions
// --------------------------------------------------------------------------------------------------------------------
// The native NVIDIA OpenCL compiler (seen on A100, driver 590.48.01) miscompiles a half8 -> float8
// __builtin_convertvector whose operand comes straight from a vector load, such as the
// ((const __global half8*)p)[0] that jitter emits for fused-op inputs: it folds the load into the widening and drops
// cvt.f32.f16 for lanes 0-5, which then hold the raw half bits as tiny denormals. Loads from global, constant and
// local memory are all affected; a half8 assembled in registers converts correctly. convert_float8 therefore widens
// lane by lane, and every helper further down widens half vectors through convert_float<w> instead of calling
// __builtin_convertvector itself, so that width 8 always takes the lane-wise path. Widths 2, 3, 4 and 16 stay on the
// builtin: vector-loaded from global, constant or local memory, or built by vloadN, they converted correctly both
// bare and through every helper below.

#undef convert_float
#define convert_float(x)   ((float)(x))
#undef convert_half
#define convert_half(x)    ((half)(x))
#undef convert_float2
#define convert_float2(x)  __builtin_convertvector((x), float2)
#undef convert_float3
#define convert_float3(x)  __builtin_convertvector((x), float3)
#undef convert_float4
#define convert_float4(x)  __builtin_convertvector((x), float4)
#undef convert_float8
static inline float8 __attribute__((overloadable)) ov_nv_convert_float8(half8 x) {
    return (float8)((float)x.s0, (float)x.s1, (float)x.s2, (float)x.s3,
                    (float)x.s4, (float)x.s5, (float)x.s6, (float)x.s7);
}
static inline float8 __attribute__((overloadable)) ov_nv_convert_float8(float8 x) {
    return x;
}
#define OV_NV_CONVERT_FLOAT8(type) \
static inline float8 __attribute__((overloadable)) ov_nv_convert_float8(type##8 x) { \
    return __builtin_convertvector(x, float8); \
}
OV_NV_CONVERT_FLOAT8(char)
OV_NV_CONVERT_FLOAT8(uchar)
OV_NV_CONVERT_FLOAT8(short)
OV_NV_CONVERT_FLOAT8(ushort)
OV_NV_CONVERT_FLOAT8(int)
OV_NV_CONVERT_FLOAT8(uint)
OV_NV_CONVERT_FLOAT8(long)
OV_NV_CONVERT_FLOAT8(ulong)
#undef OV_NV_CONVERT_FLOAT8
#define convert_float8(x) ov_nv_convert_float8(x)
#undef convert_float16
#define convert_float16(x) __builtin_convertvector((x), float16)
// Narrowing does not share the half8 problem above. On the same A100 / driver, vectors of float, double and every
// integer type at widths 2, 3, 4, 8 and 16, vector-loaded from data another work-item wrote, narrowed correctly with
// __builtin_convertvector (round to nearest even, overflow to infinity), and so did float vectors narrowed and then
// vector-stored as half at every width, so these stay on the builtin.
#undef convert_half2
#define convert_half2(x)   __builtin_convertvector((x), half2)
#undef convert_half3
#define convert_half3(x)   __builtin_convertvector((x), half3)
#undef convert_half4
#define convert_half4(x)   __builtin_convertvector((x), half4)
#undef convert_half8
#define convert_half8(x)   __builtin_convertvector((x), half8)
#undef convert_half16
#define convert_half16(x)  __builtin_convertvector((x), half16)

// --------------------------------------------------------------------------------------------------------------------
// Math built-ins
// --------------------------------------------------------------------------------------------------------------------
// Every overload computes in float and rounds the result back to half. Each arity has a per-width helper (_VEC) and a
// front macro that adds the scalar form and instantiates the helper for widths 2, 3, 4, 8 and 16.

#define OV_NV_H_UNARY_VEC(fn,w) \
static inline half##w __attribute__((overloadable)) fn(half##w x){return __builtin_convertvector(fn(convert_float##w(x)),half##w);}
#define OV_NV_H_UNARY(fn) \
static inline half __attribute__((overloadable)) fn(half x){return (half)fn((float)x);} \
OV_NV_H_UNARY_VEC(fn,2) OV_NV_H_UNARY_VEC(fn,3) OV_NV_H_UNARY_VEC(fn,4) OV_NV_H_UNARY_VEC(fn,8) OV_NV_H_UNARY_VEC(fn,16)

#define OV_NV_H_BINARY_VEC(fn,impl,w) \
static inline half##w __attribute__((overloadable)) fn(half##w a,half##w b){return __builtin_convertvector(impl(convert_float##w(a),convert_float##w(b)),half##w);}
// Mixed vector / scalar forms broadcast the scalar and reuse the vector overload.
#define OV_NV_H_BINARY_BCAST(fn,w) \
static inline half##w __attribute__((overloadable)) fn(half##w a,half b){return fn(a,(half##w)b);} \
static inline half##w __attribute__((overloadable)) fn(half a,half##w b){return fn((half##w)a,b);}
// OV_NV_H_BINARY_IMPL defines the half overloads of fn on top of the float function impl; OV_NV_H_BINARY uses fn itself.
#define OV_NV_H_BINARY_IMPL(fn,impl) \
static inline half __attribute__((overloadable)) fn(half a,half b){return (half)impl((float)a,(float)b);} \
OV_NV_H_BINARY_VEC(fn,impl,2) OV_NV_H_BINARY_VEC(fn,impl,3) OV_NV_H_BINARY_VEC(fn,impl,4) OV_NV_H_BINARY_VEC(fn,impl,8) \
OV_NV_H_BINARY_VEC(fn,impl,16) \
OV_NV_H_BINARY_BCAST(fn,2) OV_NV_H_BINARY_BCAST(fn,3) OV_NV_H_BINARY_BCAST(fn,4) OV_NV_H_BINARY_BCAST(fn,8) OV_NV_H_BINARY_BCAST(fn,16)
#define OV_NV_H_BINARY(fn) OV_NV_H_BINARY_IMPL(fn,fn)

// pow on the native NVIDIA compiler is not correctly rounded even when the exact result is representable in float:
// pow(114.0f, 2.0f) returns 12996.001f on A100 (driver 590.48.01). Rounding to half turns that float ulp into a half
// ulp whenever the exact result sits halfway between two halves (12996 -> 13000 instead of 12992), so half pow with an
// integral exponent goes through pown, which gave the correctly rounded half for every finite half base and exponents
// -3..6 on the same device. The bound keeps the int conversion defined and still covers every finite half exponent,
// half pow being the only caller.
// An exponent of 0.5 has the same problem (pow(15.992188f, 0.5f) rounds to 4 instead of 3.998047) and goes through
// sqrt, which gave the correctly rounded half for every half base. fabs turns sqrt(-0) = -0 into the +0 pow returns,
// and -inf is mapped to the +inf pow returns, where sqrt gives NaN.
// Exponents of -1 and -0.5 are taken out for speed: pown(a, -1) is slower than pow, and both are about ten times
// slower than 1 / a and rsqrt on the same device. Exponent -1 comes from every non-constant float Divide, which
// ConvertDivide turns into a multiply by pow(b, -1); -0.5 comes from RMS / MVN patterns that are not fused. Both forms
// gave the correctly rounded half for every half base; fabs turns rsqrt(-0) = -inf into the +inf pow returns, and -inf
// is mapped to the +0 pow returns, where rsqrt gives NaN.
// A constant exponent folds all these tests away at compile time.
static inline float __attribute__((overloadable)) ov_nv_pow_f(float a, float b) {
    if (b == 0.5f)
        return (a == -INFINITY) ? INFINITY : fabs(sqrt(a));
    if (b == -0.5f)
        return (a == -INFINITY) ? 0.0f : fabs(rsqrt(a));
    if (b == -1.0f)
        return 1.0f / a;
    return (b == rint(b) && fabs(b) <= 65504.0f) ? pown(a, (int)b) : pow(a, b);
}
// Exponents of -0.5 and -1 shared by every lane take the fast forms. Otherwise lanes with a non-integral exponent pass
// 0 to pown and take the pow result through select, and lanes with an exponent of 0.5 then take the sqrt result.
#define OV_NV_POW_F_VEC(w) \
static inline float##w __attribute__((overloadable)) ov_nv_pow_f(float##w a, float##w b) { \
    if (all(b == -0.5f)) \
        return select(fabs(rsqrt(a)), (float##w)0.0f, isequal(a, (float##w)(-INFINITY))); \
    if (all(b == -1.0f)) \
        return 1.0f / a; \
    int##w m = isequal(b, rint(b)) & islessequal(fabs(b), (float##w)65504.0f); \
    float##w r = select(pow(a, b), pown(a, __builtin_convertvector(select((float##w)0.0f, b, m), int##w)), m); \
    float##w s = select(fabs(sqrt(a)), (float##w)INFINITY, isequal(a, (float##w)(-INFINITY))); \
    return select(r, s, isequal(b, (float##w)0.5f)); \
}
OV_NV_POW_F_VEC(2) OV_NV_POW_F_VEC(3) OV_NV_POW_F_VEC(4) OV_NV_POW_F_VEC(8) OV_NV_POW_F_VEC(16)
#undef OV_NV_POW_F_VEC

#define OV_NV_H_TERNARY_VEC(fn,w) \
static inline half##w __attribute__((overloadable)) fn(half##w a,half##w b,half##w c){return __builtin_convertvector(fn(convert_float##w(a),convert_float##w(b),convert_float##w(c)),half##w);}
#define OV_NV_H_TERNARY(fn) \
static inline half __attribute__((overloadable)) fn(half a,half b,half c){return (half)fn((float)a,(float)b,(float)c);} \
OV_NV_H_TERNARY_VEC(fn,2) OV_NV_H_TERNARY_VEC(fn,3) OV_NV_H_TERNARY_VEC(fn,4) OV_NV_H_TERNARY_VEC(fn,8) OV_NV_H_TERNARY_VEC(fn,16)

OV_NV_H_UNARY(acos)   OV_NV_H_UNARY(acosh)  OV_NV_H_UNARY(asin)   OV_NV_H_UNARY(asinh)
OV_NV_H_UNARY(atan)   OV_NV_H_UNARY(atanh)  OV_NV_H_UNARY(cbrt)   OV_NV_H_UNARY(ceil)
OV_NV_H_UNARY(cos)    OV_NV_H_UNARY(cosh)   OV_NV_H_UNARY(erf)    OV_NV_H_UNARY(erfc)
OV_NV_H_UNARY(exp)    OV_NV_H_UNARY(exp2)   OV_NV_H_UNARY(exp10)  OV_NV_H_UNARY(expm1)
OV_NV_H_UNARY(fabs)   OV_NV_H_UNARY(floor)  OV_NV_H_UNARY(lgamma) OV_NV_H_UNARY(log)
OV_NV_H_UNARY(log2)   OV_NV_H_UNARY(log10)  OV_NV_H_UNARY(log1p)  OV_NV_H_UNARY(rint)
OV_NV_H_UNARY(round)  OV_NV_H_UNARY(rsqrt)  OV_NV_H_UNARY(sin)    OV_NV_H_UNARY(sinh)
OV_NV_H_UNARY(sqrt)   OV_NV_H_UNARY(tan)    OV_NV_H_UNARY(tanh)   OV_NV_H_UNARY(tgamma)
OV_NV_H_UNARY(trunc)  OV_NV_H_UNARY(degrees) OV_NV_H_UNARY(radians)
OV_NV_H_UNARY(native_cos)  OV_NV_H_UNARY(native_exp)  OV_NV_H_UNARY(native_exp2)
OV_NV_H_UNARY(native_exp10) OV_NV_H_UNARY(native_log) OV_NV_H_UNARY(native_log2)
OV_NV_H_UNARY(native_log10) OV_NV_H_UNARY(native_recip) OV_NV_H_UNARY(native_rsqrt)
OV_NV_H_UNARY(native_sin)  OV_NV_H_UNARY(native_sqrt) OV_NV_H_UNARY(native_tan)
OV_NV_H_BINARY(atan2)     OV_NV_H_BINARY(copysign) OV_NV_H_BINARY(fdim)
OV_NV_H_BINARY(fmax)      OV_NV_H_BINARY(fmin)     OV_NV_H_BINARY(fmod)
OV_NV_H_BINARY(hypot)     OV_NV_H_BINARY(max)      OV_NV_H_BINARY(min)
OV_NV_H_BINARY(maxmag)    OV_NV_H_BINARY(minmag)
OV_NV_H_BINARY_IMPL(pow, ov_nv_pow_f) OV_NV_H_BINARY(powr) OV_NV_H_BINARY(remainder)
OV_NV_H_BINARY(step)      OV_NV_H_BINARY(native_divide)
OV_NV_H_TERNARY(mad)
OV_NV_H_TERNARY(fma)
OV_NV_H_TERNARY(clamp)
OV_NV_H_TERNARY(mix)

#undef OV_NV_H_UNARY_VEC
#undef OV_NV_H_UNARY
#undef OV_NV_H_BINARY_VEC
#undef OV_NV_H_BINARY_BCAST
#undef OV_NV_H_BINARY
#undef OV_NV_H_BINARY_IMPL
#undef OV_NV_H_TERNARY_VEC
#undef OV_NV_H_TERNARY

// OpenCL select() uses a nonzero condition for scalar arguments; the MSB rule
// applies only when the arguments are vectors.
#define OV_NV_H_SELECT(mask_type) \
static inline half __attribute__((overloadable)) select(half a, half b, mask_type c){return c ? b : a;}
OV_NV_H_SELECT(char)
OV_NV_H_SELECT(short)
OV_NV_H_SELECT(int)
OV_NV_H_SELECT(long)
#undef OV_NV_H_SELECT

#undef native_powr
#define native_powr(x, y) native_exp2((y) * native_log2(fabs(x)))

// --------------------------------------------------------------------------------------------------------------------
// Integer conversions
// --------------------------------------------------------------------------------------------------------------------

// No plain half overloads for char, uchar, short and ushort, scalar or vector: the out-of-range section below turns
// those names into macros that take a half argument directly, so a function of the same name would never be called.
static inline int    __attribute__((overloadable)) convert_int     (half x){return (int   )(float)(x);}
static inline uint   __attribute__((overloadable)) convert_uint    (half x){return (uint  )(float)(x);}
static inline long   __attribute__((overloadable)) convert_long    (half x){return (long  )(float)(x);}
static inline ulong  __attribute__((overloadable)) convert_ulong   (half x){return (ulong )(float)(x);}
static inline char   __attribute__((overloadable)) convert_char_sat  (half x){return convert_char_sat  ((float)x);}
static inline uchar  __attribute__((overloadable)) convert_uchar_sat (half x){return convert_uchar_sat ((float)x);}
static inline short  __attribute__((overloadable)) convert_short_sat (half x){return convert_short_sat ((float)x);}
static inline ushort __attribute__((overloadable)) convert_ushort_sat(half x){return convert_ushort_sat((float)x);}

// Half vectors reach the integer types through float, widened by convert_float<w> (see the half <-> float section).
#define OV_NV_CONV_INT_VEC(fn, type, w) \
static inline type##w __attribute__((overloadable)) fn##w(half##w x){return __builtin_convertvector(convert_float##w(x),type##w);}
#define OV_NV_CONV_INT(fn, type) \
OV_NV_CONV_INT_VEC(fn, type, 2) OV_NV_CONV_INT_VEC(fn, type, 3) OV_NV_CONV_INT_VEC(fn, type, 4) \
OV_NV_CONV_INT_VEC(fn, type, 8) OV_NV_CONV_INT_VEC(fn, type, 16)
// The rounding families (_rte, _rtp, _sat_rte) differ only in the suffix, which is passed as sfx. Each converts to
// float and hands off to the float overload of the same name.
#define OV_NV_CONV_INT_SFX_VEC(fn, type, sfx, w) \
static inline type##w __attribute__((overloadable)) fn##w##sfx(half##w x){return fn##w##sfx(convert_float##w(x));}
#define OV_NV_CONV_INT_SFX(fn, type, sfx) \
static inline type __attribute__((overloadable)) fn##sfx(half x){return fn##sfx((float)x);} \
OV_NV_CONV_INT_SFX_VEC(fn, type, sfx, 2) OV_NV_CONV_INT_SFX_VEC(fn, type, sfx, 3) \
OV_NV_CONV_INT_SFX_VEC(fn, type, sfx, 4) OV_NV_CONV_INT_SFX_VEC(fn, type, sfx, 8) \
OV_NV_CONV_INT_SFX_VEC(fn, type, sfx, 16)
#define OV_NV_CONV_INT_ROUNDING(fn, type) \
OV_NV_CONV_INT_SFX(fn, type, _rte) OV_NV_CONV_INT_SFX(fn, type, _rtp) OV_NV_CONV_INT_SFX(fn, type, _sat_rte)
#define OV_NV_CONV_INT_ALL(fn, type) \
OV_NV_CONV_INT(fn, type) OV_NV_CONV_INT_ROUNDING(fn, type)
// cl_kernel.h declares convert_<type>[_rte|_rtp|_sat_rte] for every source type except half, so each rounding
// family is filled in below for all eight integer destinations, scalar forms included. Leaving a family at
// char/uchar only breaks any kernel that rounds a half vector into a wider destination, such as the
// convert_int8_rte in dynamic_quantize_gpu_opt.cl. The plain family is only needed for the four 32- and 64-bit
// destinations; see above for the narrow ones.
OV_NV_CONV_INT_ROUNDING(convert_char,   char)
OV_NV_CONV_INT_ROUNDING(convert_uchar,  uchar)
OV_NV_CONV_INT_ROUNDING(convert_short,  short)
OV_NV_CONV_INT_ROUNDING(convert_ushort, ushort)
OV_NV_CONV_INT_ALL(convert_int,    int)
OV_NV_CONV_INT_ALL(convert_uint,   uint)
OV_NV_CONV_INT_ALL(convert_long,   long)
OV_NV_CONV_INT_ALL(convert_ulong,  ulong)
#undef OV_NV_CONV_INT_VEC
#undef OV_NV_CONV_INT
#undef OV_NV_CONV_INT_SFX_VEC
#undef OV_NV_CONV_INT_SFX
#undef OV_NV_CONV_INT_ROUNDING
#undef OV_NV_CONV_INT_ALL

// --------------------------------------------------------------------------------------------------------------------
// Out-of-range float to integer conversions
// --------------------------------------------------------------------------------------------------------------------
// convert_<type>() without _sat is undefined by the OpenCL spec when the value does not fit the destination, and the
// NVIDIA implementation clamps the whole way down: convert_char(4404.0f) gives 127 and convert_uchar(-127.0f) gives
// 0. Intel converts in two steps instead - float to the 32-bit integer of the *same signedness* as the destination,
// then a truncation to the narrow type - and kernels are written against that result.
// The two steps behave differently at each end of the range, so the signed and the unsigned forms need different
// emulations:
//   - char/short go through int, where the float conversion keeps the value and the truncation keeps the low bits:
//     convert_char(4404.0f) is 52. reduce_gpu_b_fs_yx_fsv16 accumulates in int, converts to ACTIVATION_TYPE float
//     and converts that to its i8 output, so a sum of 4404 has to come out as 52.
//   - uchar/ushort go through uint, where a negative float has no representation and clamps to zero before the
//     truncation ever runs: convert_uchar(-127.0f) is 0. quantize writes its output range low bound straight through
//     TO_OUTPUT_TYPE, so an out_low of -127 has to come out as 0 and not as the 129 that int would leave behind.
// Positive overflow still wraps in both forms - convert_uchar(4404.0f) is 52 - and in-range values are unaffected:
// the float still truncates toward zero. Integer sources are unaffected too, since int and uint truncate to the same
// low bits.
// These have to be macros: the scalar forms are already declared non-static by cl_kernel.h and the vector forms are
// macros there, so neither can be overridden with a definition of the same signature.
#undef convert_char
#define convert_char(x) ((char)(int)(x))
#undef convert_uchar
#define convert_uchar(x) ((uchar)(uint)(x))
#undef convert_short
#define convert_short(x) ((short)(int)(x))
#undef convert_ushort
#define convert_ushort(x) ((ushort)(uint)(x))
#undef convert_char2
#define convert_char2(x) __builtin_convertvector(__builtin_convertvector((x), int2), char2)
#undef convert_uchar2
#define convert_uchar2(x) __builtin_convertvector(__builtin_convertvector((x), uint2), uchar2)
#undef convert_short2
#define convert_short2(x) __builtin_convertvector(__builtin_convertvector((x), int2), short2)
#undef convert_ushort2
#define convert_ushort2(x) __builtin_convertvector(__builtin_convertvector((x), uint2), ushort2)
#undef convert_char3
#define convert_char3(x) __builtin_convertvector(__builtin_convertvector((x), int3), char3)
#undef convert_uchar3
#define convert_uchar3(x) __builtin_convertvector(__builtin_convertvector((x), uint3), uchar3)
#undef convert_short3
#define convert_short3(x) __builtin_convertvector(__builtin_convertvector((x), int3), short3)
#undef convert_ushort3
#define convert_ushort3(x) __builtin_convertvector(__builtin_convertvector((x), uint3), ushort3)
#undef convert_char4
#define convert_char4(x) __builtin_convertvector(__builtin_convertvector((x), int4), char4)
#undef convert_uchar4
#define convert_uchar4(x) __builtin_convertvector(__builtin_convertvector((x), uint4), uchar4)
#undef convert_short4
#define convert_short4(x) __builtin_convertvector(__builtin_convertvector((x), int4), short4)
#undef convert_ushort4
#define convert_ushort4(x) __builtin_convertvector(__builtin_convertvector((x), uint4), ushort4)
#undef convert_char8
#define convert_char8(x) __builtin_convertvector(__builtin_convertvector((x), int8), char8)
#undef convert_uchar8
#define convert_uchar8(x) __builtin_convertvector(__builtin_convertvector((x), uint8), uchar8)
#undef convert_short8
#define convert_short8(x) __builtin_convertvector(__builtin_convertvector((x), int8), short8)
#undef convert_ushort8
#define convert_ushort8(x) __builtin_convertvector(__builtin_convertvector((x), uint8), ushort8)
#undef convert_char16
#define convert_char16(x) __builtin_convertvector(__builtin_convertvector((x), int16), char16)
#undef convert_uchar16
#define convert_uchar16(x) __builtin_convertvector(__builtin_convertvector((x), uint16), uchar16)
#undef convert_short16
#define convert_short16(x) __builtin_convertvector(__builtin_convertvector((x), int16), short16)
#undef convert_ushort16
#define convert_ushort16(x) __builtin_convertvector(__builtin_convertvector((x), uint16), ushort16)

// --------------------------------------------------------------------------------------------------------------------
// Reinterpretation (as_*)
// --------------------------------------------------------------------------------------------------------------------

#define OV_NV_ASHALF(w) \
static inline half##w   __attribute__((overloadable)) as_half##w  (ushort##w x){return __builtin_astype(x, half##w  );} \
static inline half##w   __attribute__((overloadable)) as_half##w  (short##w  x){return __builtin_astype(x, half##w  );} \
static inline half##w   __attribute__((overloadable)) as_half##w  (half##w   x){return x;} \
static inline ushort##w __attribute__((overloadable)) as_ushort##w(half##w   x){return __builtin_astype(x, ushort##w);} \
static inline short##w  __attribute__((overloadable)) as_short##w (half##w   x){return __builtin_astype(x, short##w );}
static inline half   __attribute__((overloadable)) as_half  (ushort x){return __builtin_astype(x, half  );}
static inline half   __attribute__((overloadable)) as_half  (short  x){return __builtin_astype(x, half  );}
static inline half   __attribute__((overloadable)) as_half  (half   x){return x;}
static inline ushort __attribute__((overloadable)) as_ushort(half   x){return __builtin_astype(x, ushort);}
static inline short  __attribute__((overloadable)) as_short (half   x){return __builtin_astype(x, short );}
OV_NV_ASHALF(2) OV_NV_ASHALF(3) OV_NV_ASHALF(4) OV_NV_ASHALF(8) OV_NV_ASHALF(16)
#undef OV_NV_ASHALF

// Provide overloads to reinterpret two packed half values as one 32-bit scalar.
static inline float __attribute__((overloadable)) as_float(half2 x) {
    return __builtin_astype(x, float);
}
static inline uint __attribute__((overloadable)) as_uint(half2 x) {
    return __builtin_astype(x, uint);
}
static inline int __attribute__((overloadable)) as_int(half2 x) {
    return __builtin_astype(x, int);
}
// Provide overloads to reinterpret a packed 32-bit value as two 16-bit halves
static inline half2 __attribute__((overloadable)) as_half2(uint x){
    ushort2 u = (ushort2)((ushort)(x & 0xffffu),(ushort)((x >> 16) & 0xffffu));
    return as_half2(u);
}
static inline half2 __attribute__((overloadable)) as_half2(int x){ return as_half2((uint)x); }

// Provide overload to reinterpret 8x uint32 read (uint8) as 16x half (half16).
// Each uint32 contains two 16-bit half bitpatterns: low 16 bits then high 16 bits.
static inline half16 __attribute__((overloadable)) as_half16(uint8 x) {
    ushort16 u = (ushort16)(
        (ushort)(x.s0 & 0xffffu), (ushort)((x.s0 >> 16) & 0xffffu),
        (ushort)(x.s1 & 0xffffu), (ushort)((x.s1 >> 16) & 0xffffu),
        (ushort)(x.s2 & 0xffffu), (ushort)((x.s2 >> 16) & 0xffffu),
        (ushort)(x.s3 & 0xffffu), (ushort)((x.s3 >> 16) & 0xffffu),
        (ushort)(x.s4 & 0xffffu), (ushort)((x.s4 >> 16) & 0xffffu),
        (ushort)(x.s5 & 0xffffu), (ushort)((x.s5 >> 16) & 0xffffu),
        (ushort)(x.s6 & 0xffffu), (ushort)((x.s6 >> 16) & 0xffffu),
        (ushort)(x.s7 & 0xffffu), (ushort)((x.s7 >> 16) & 0xffffu)
    );
    return as_half16(u);
}

// --------------------------------------------------------------------------------------------------------------------
// vloadN / vstoreN
// --------------------------------------------------------------------------------------------------------------------

#define OV_NV_VLOAD(as) \
static inline half2  __attribute__((overloadable)) vload2 (size_t o,const as half* p){return (half2 )(p[o*2+0],p[o*2+1]);} \
static inline half3  __attribute__((overloadable)) vload3 (size_t o,const as half* p){return (half3 )(p[o*3+0],p[o*3+1],p[o*3+2]);} \
static inline half4  __attribute__((overloadable)) vload4 (size_t o,const as half* p){return (half4 )(p[o*4+0],p[o*4+1],p[o*4+2],p[o*4+3]);} \
static inline half8  __attribute__((overloadable)) vload8 (size_t o,const as half* p){return (half8 )(p[o*8+0],p[o*8+1],p[o*8+2],p[o*8+3],p[o*8+4],p[o*8+5],p[o*8+6],p[o*8+7]);} \
static inline half16 __attribute__((overloadable)) vload16(size_t o,const as half* p){return (half16)(p[o*16+0],p[o*16+1],p[o*16+2],p[o*16+3],p[o*16+4],p[o*16+5],p[o*16+6],p[o*16+7],p[o*16+8],p[o*16+9],p[o*16+10],p[o*16+11],p[o*16+12],p[o*16+13],p[o*16+14],p[o*16+15]);}
OV_NV_VLOAD(__global)
OV_NV_VLOAD(__constant)
OV_NV_VLOAD(__local)
#undef OV_NV_VLOAD

#define OV_NV_VSTORE(as) \
static inline void __attribute__((overloadable)) vstore2 (half2  v,size_t o,as half* p){p[o*2+0]=v.s0;p[o*2+1]=v.s1;} \
static inline void __attribute__((overloadable)) vstore3 (half3  v,size_t o,as half* p){p[o*3+0]=v.s0;p[o*3+1]=v.s1;p[o*3+2]=v.s2;} \
static inline void __attribute__((overloadable)) vstore4 (half4  v,size_t o,as half* p){p[o*4+0]=v.s0;p[o*4+1]=v.s1;p[o*4+2]=v.s2;p[o*4+3]=v.s3;} \
static inline void __attribute__((overloadable)) vstore8 (half8  v,size_t o,as half* p){p[o*8+0]=v.s0;p[o*8+1]=v.s1;p[o*8+2]=v.s2;p[o*8+3]=v.s3;p[o*8+4]=v.s4;p[o*8+5]=v.s5;p[o*8+6]=v.s6;p[o*8+7]=v.s7;} \
static inline void __attribute__((overloadable)) vstore16(half16 v,size_t o,as half* p){p[o*16+0]=v.s0;p[o*16+1]=v.s1;p[o*16+2]=v.s2;p[o*16+3]=v.s3;p[o*16+4]=v.s4;p[o*16+5]=v.s5;p[o*16+6]=v.s6;p[o*16+7]=v.s7;p[o*16+8]=v.s8;p[o*16+9]=v.s9;p[o*16+10]=v.sa;p[o*16+11]=v.sb;p[o*16+12]=v.sc;p[o*16+13]=v.sd;p[o*16+14]=v.se;p[o*16+15]=v.sf;}
OV_NV_VSTORE(__global)
#undef OV_NV_VSTORE

// --------------------------------------------------------------------------------------------------------------------
// Relational built-ins
// --------------------------------------------------------------------------------------------------------------------
// NVIDIA OpenCL headers expose relational built-ins for float and double, but
// not for half, so the overloads below fill that gap.
//
// A relational built-in taking halfN returns shortN, not intN: the result feeds
// select() and the vector form of ?:, both of which require the condition to have
// the same element size as the operands. Returning intN makes the compiler reject
// 'half2 r = cond ? a : b' and assignments to the short vectors that callers such
// as detection_output_common.cl (CMP_TYPE4) declare.
#define OV_NV_H_RELATIONAL_VEC(fn,w) \
static inline short##w __attribute__((overloadable)) fn(half##w a,half##w b){return convert_short##w(fn(convert_float##w(a),convert_float##w(b)));}
#define OV_NV_H_RELATIONAL(fn) \
OV_NV_H_RELATIONAL_VEC(fn,2) OV_NV_H_RELATIONAL_VEC(fn,3) OV_NV_H_RELATIONAL_VEC(fn,4) OV_NV_H_RELATIONAL_VEC(fn,8) OV_NV_H_RELATIONAL_VEC(fn,16)
OV_NV_H_RELATIONAL(isgreater)
OV_NV_H_RELATIONAL(islessequal)
#undef OV_NV_H_RELATIONAL_VEC
#undef OV_NV_H_RELATIONAL

static inline bool __attribute__((overloadable)) isinf(half x) {
    ushort h = as_ushort(x);
    return (h & (ushort)0x7FFF) == (ushort)0x7C00;
}

// The vector forms are reached through the activation jit for RELU_NEGATIVE_SLOPE,
// which expands to '(isinf(m)) ? ... : ...' with m typed as the vectorised slope.
// Same element-size rule as above, hence shortN.
#define OV_NV_H_ISINF(w) \
static inline short##w __attribute__((overloadable)) isinf(half##w x){return convert_short##w(isinf(convert_float##w(x)));}
OV_NV_H_ISINF(2) OV_NV_H_ISINF(3) OV_NV_H_ISINF(4) OV_NV_H_ISINF(8) OV_NV_H_ISINF(16)
#undef OV_NV_H_ISINF

// Reached through the eltwise jit for MIN / MAX, which propagates NaN with
// 'isnan((ACCUMULATOR_TYPE)x) ? ...'. Without a half overload the call is ambiguous
// between the float / double / vector candidates in cl_kernel.h.
// The scalar form returns int as the OpenCL spec requires; the vector forms return
// shortN by the element-size rule above.
static inline int __attribute__((overloadable)) isnan(half x) {
    ushort h = as_ushort(x);
    return (h & (ushort)0x7FFF) > (ushort)0x7C00;
}
#define OV_NV_H_ISNAN(w) \
static inline short##w __attribute__((overloadable)) isnan(half##w x){return convert_short##w(isnan(convert_float##w(x)));}
OV_NV_H_ISNAN(2) OV_NV_H_ISNAN(3) OV_NV_H_ISNAN(4) OV_NV_H_ISNAN(8) OV_NV_H_ISNAN(16)
#undef OV_NV_H_ISNAN

// --------------------------------------------------------------------------------------------------------------------
// Atomics on half (cl_ext_float_atomics, __opencl_c_ext_fp16_global_atomic_load_store)
// --------------------------------------------------------------------------------------------------------------------
// The NVIDIA compiler has neither atomic_half nor the fp16 atomic_load / atomic_store / atomic_exchange that
// cl_ext_float_atomics adds. istft_ref.cl declares its f16 accumulation helper for every output type, #errors when the
// feature macro is missing and accumulates f16 output with atomic_exchange, so without these even its f32 kernels
// failed to build. PTX has no 16-bit atom.exch, so the exchange is a loop on the 16-bit atom.cas.b16 (sm_70 and later).
// Only the default-order forms are provided: sequentially consistent at device scope, done by placing fence.sc.gpu on
// both sides of a relaxed access, which is stronger than required. The _explicit forms are left out on purpose, so a
// caller asking for them fails to compile rather than getting different semantics. atomic_half is a plain typedef of
// half, so unlike a real atomic type it does not stop non-atomic accesses. The functions are static inline and reach
// only the PTX of kernels that call them.
typedef half atomic_half;
#ifndef __opencl_c_ext_fp16_global_atomic_load_store
#define __opencl_c_ext_fp16_global_atomic_load_store 1
#endif

static inline ushort _nv_atomic_cas_u16(volatile atomic_half* p, ushort expected, ushort desired) {
    ushort prev;
    __asm__ volatile("atom.relaxed.gpu.cas.b16 %0, [%1], %2, %3;"
                     : "=h"(prev) : "l"(p), "h"(expected), "h"(desired) : "memory");
    return prev;
}

static inline void _nv_fence_sc_gpu(void) {
    __asm__ volatile("fence.sc.gpu;" ::: "memory");
}

static inline half __attribute__((overloadable)) atomic_exchange(volatile atomic_half* p, half desired) {
    const ushort d = as_ushort(desired);
    _nv_fence_sc_gpu();
    ushort expected = *(volatile ushort*)p;
    ushort prev;
    while ((prev = _nv_atomic_cas_u16(p, expected, d)) != expected)
        expected = prev;
    _nv_fence_sc_gpu();
    return as_half(prev);
}

static inline half __attribute__((overloadable)) atomic_load(volatile atomic_half* p) {
    ushort v;
    _nv_fence_sc_gpu();
    __asm__ volatile("ld.relaxed.gpu.b16 %0, [%1];" : "=h"(v) : "l"(p) : "memory");
    _nv_fence_sc_gpu();
    return as_half(v);
}

static inline void __attribute__((overloadable)) atomic_store(volatile atomic_half* p, half desired) {
    (void)atomic_exchange(p, desired);
}

#endif // __NV_CL_C_VERSION
