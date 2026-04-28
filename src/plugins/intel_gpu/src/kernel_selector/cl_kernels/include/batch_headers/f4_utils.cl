// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// TODO: Replace `_intel_convert*` with bultins when ready, current implementations are copied from XeTLA:

#define FP4_MASK       0x0F
#define FP4_SHIFT      4
#define FP4_HIGH_MASK  0xF0

uchar _intel_convert_f16_to_f4(half val) {
    half val_fp16 = val;
    ushort *p = (ushort *)&val_fp16;
    ushort src = p[0];
    const ushort src_exp_size = 5;
    const ushort src_mant_size = 10;
    const ushort src_exp_bias = (1 << (src_exp_size - 1)) - 1;
    const ushort src_exp_mask = (1 << src_exp_size) - 1;
    const ushort src_mant_mask = (1 << src_mant_size) - 1;
    const short max_exp_unbiased = 2;
    const short min_exp_unbiased = 0;
    const short exp_bias = 1;
    const ushort exp_size = 2;
    const ushort mant_size = 1;
    const uchar  max_val = 0x7;

    ushort src_sign = src >> (src_exp_size + src_mant_size);
    ushort src_exp = (src >> src_mant_size) & src_exp_mask;
    short src_exp_unbiased = src_exp - src_exp_bias;
    ushort src_mant = src & src_mant_mask;

    bool is_src_inf_nan = src_exp == 0x1f;
    bool is_overflow = (src_exp_unbiased > max_exp_unbiased)
            || ((src_exp_unbiased == max_exp_unbiased) && (src_mant > 0x0340));
    bool is_zero = (src_exp_unbiased < (min_exp_unbiased - mant_size));
    bool is_denorm = (src_exp_unbiased < min_exp_unbiased) && (!is_zero);

    uchar  dst_val;
    if (is_src_inf_nan || is_overflow) {
        dst_val = max_val;
    } else if (is_zero) {
        dst_val = 0;
        if (src_exp == 0xD && src_mant & 0x3FF) // if larger then 0.25 round to 0.5
            dst_val = 0x1;
    } else if (is_denorm) {
        dst_val = 0x1;
        if (src_exp == 0xE && src_mant & 0x200) // if larger then 0.75 round to 1.0
            dst_val = 0x2;
    } else {
        ushort tail_size = src_mant_size - mant_size;
        bool sticky_flag = (src_mant & ((1 << (tail_size - 1)) - 1)) != 0;
        bool lsb_bit = src_mant & (1 << tail_size);
        bool rnd_bit = src_mant & (1 << (tail_size - 1));
        bool carry = (lsb_bit && rnd_bit) || (rnd_bit && sticky_flag);
        ushort src_m = (src_mant >> tail_size) + carry;
        ushort src_e = src_exp_unbiased + exp_bias;
        dst_val = (src_e << mant_size) + src_m;
        if(dst_val > max_val){
            dst_val = max_val;
        }
    }
    return (src_sign << (exp_size + mant_size) | dst_val) & FP4_MASK;
}

half _intel_convert_fp4_to_f16(uchar val){
        static const ushort LUT[16] = {
                0x0000, 0x3800, 0x3c00, 0x3e00, 0x4000, 0x4200, 0x4400, 0x4600, 
                0x8000, 0xb800, 0xbc00, 0xbe00, 0xc000, 0xc200, 0xc400, 0xc600
        };

        ushort idx = val & FP4_MASK;
        return as_half(LUT[idx]);
}

static inline uchar pack_nibbles(uchar low, uchar high) {
    return ((high << FP4_SHIFT) & FP4_HIGH_MASK) | (low & FP4_MASK);
}

static inline uchar unpack_nibble_low(uchar packed) {
    return packed & FP4_MASK;
}

static inline uchar unpack_nibble_high(uchar packed) {
    return (packed >> FP4_SHIFT) & FP4_MASK;
}

static inline uchar2 unpack_nibbles(uchar packed) {
    return (uchar2)(unpack_nibble_low(packed), unpack_nibble_high(packed));
}

static inline uchar pack_half_pair_to_fp4(half low, half high) {
    return pack_nibbles(_intel_convert_f16_to_f4(low), _intel_convert_f16_to_f4(high));
}

static inline uchar pack_float_pair_to_fp4(float low, float high) {
    return pack_half_pair_to_fp4((half)low, (half)high);
}

static inline half unpack_fp4_low_to_half(uchar packed) {
    return _intel_convert_fp4_to_f16(unpack_nibble_low(packed));
}

static inline half unpack_fp4_high_to_half(uchar packed) {
    return _intel_convert_fp4_to_f16(unpack_nibble_high(packed));
}

static inline half2 unpack_fp4_to_half_pair(uchar packed) {
    return (half2)(unpack_fp4_low_to_half(packed), unpack_fp4_high_to_half(packed));
}

typedef struct fp4e2m1_t { uchar data; } fp4e2m1_t;  // f4
typedef struct fp4e2m1_t1 { uchar data; } fp4e2m1_t1;
typedef struct fp4e2m1_t2 { uchar data; } fp4e2m1_t2;
typedef struct fp4e2m1_t3 { uchar2 data; } fp4e2m1_t3;
typedef struct fp4e2m1_t4 { uchar2 data; } fp4e2m1_t4;
typedef struct fp4e2m1_t8 { uchar4 data; } fp4e2m1_t8;
typedef struct fp4e2m1_t16 { uchar8 data; } fp4e2m1_t16;

half __attribute__((overloadable)) _convert_half(fp4e2m1_t val) {
    return unpack_fp4_low_to_half(val.data);
}
half __attribute__((overloadable)) _convert_half(fp4e2m1_t1 val) {
    return unpack_fp4_low_to_half(val.data);
}
half2 __attribute__((overloadable)) _convert_half2(fp4e2m1_t2 val) {
    return unpack_fp4_to_half_pair(val.data);
}
half3 __attribute__((overloadable)) _convert_half3(fp4e2m1_t3 val) {
    return (half3)(unpack_fp4_low_to_half(val.data.s0), unpack_fp4_high_to_half(val.data.s0), 
                   unpack_fp4_low_to_half(val.data.s1));
}
half4 __attribute__((overloadable)) _convert_half4(fp4e2m1_t4 val) {
    return (half4)(
        unpack_fp4_to_half_pair(val.data.s0),
        unpack_fp4_to_half_pair(val.data.s1)
    );
}
half8 __attribute__((overloadable)) _convert_half8(fp4e2m1_t8 val) {
    return (half8)(
        unpack_fp4_to_half_pair(val.data.s0),
        unpack_fp4_to_half_pair(val.data.s1),
        unpack_fp4_to_half_pair(val.data.s2),
        unpack_fp4_to_half_pair(val.data.s3)
    );
}
half16 __attribute__((overloadable)) _convert_half16(fp4e2m1_t16 val) {
    return (half16)(
        unpack_fp4_to_half_pair(val.data.s0), unpack_fp4_to_half_pair(val.data.s1),
        unpack_fp4_to_half_pair(val.data.s2), unpack_fp4_to_half_pair(val.data.s3),
        unpack_fp4_to_half_pair(val.data.s4), unpack_fp4_to_half_pair(val.data.s5),
        unpack_fp4_to_half_pair(val.data.s6), unpack_fp4_to_half_pair(val.data.s7)
    );
}

float __attribute__((overloadable)) _convert_float(fp4e2m1_t val) {
    return (float)_convert_half(val);
}
float __attribute__((overloadable)) _convert_float(fp4e2m1_t1 val) {
    return (float)_convert_half(val);
}
float2 __attribute__((overloadable)) _convert_float2(fp4e2m1_t2 val) {
    return _convert_float2(_convert_half2(val));
}
float3 __attribute__((overloadable)) _convert_float3(fp4e2m1_t3 val) {
    return _convert_float3(_convert_half3(val));
}
float4 __attribute__((overloadable)) _convert_float4(fp4e2m1_t4 val) {
    return _convert_float4(_convert_half4(val));
}
float8 __attribute__((overloadable)) _convert_float8(fp4e2m1_t8 val) {
    return _convert_float8(_convert_half8(val));
}
float16 __attribute__((overloadable)) _convert_float16(fp4e2m1_t16 val) {
    return _convert_float16(_convert_half16(val));
}


fp4e2m1_t __attribute__((overloadable)) _convert_fp4e2m1_t(half val) {
    fp4e2m1_t res;
    res.data = _intel_convert_f16_to_f4(val);
    return res;
}
fp4e2m1_t1 __attribute__((overloadable)) _convert_fp4e2m1_t1(half val[1]) {
    fp4e2m1_t1 res;
    res.data = _intel_convert_f16_to_f4(val[0]);
    return res;
}
fp4e2m1_t2 __attribute__((overloadable)) _convert_fp4e2m1_t2(half2 val) {
    fp4e2m1_t2 res;
    res.data = pack_half_pair_to_fp4(val.x, val.y);
    return res;
}
fp4e2m1_t3 __attribute__((overloadable)) _convert_fp4e2m1_t3(half3 val) {
    fp4e2m1_t3 res;
    res.data.s0 = pack_half_pair_to_fp4(val.x, val.y);
    res.data.s1 = _intel_convert_f16_to_f4(val.z);
    return res;
}
fp4e2m1_t4 __attribute__((overloadable)) _convert_fp4e2m1_t4(half4 val) {
    fp4e2m1_t4 res;
    res.data.s0 = pack_half_pair_to_fp4(val.x, val.y);
    res.data.s1 = pack_half_pair_to_fp4(val.z, val.w);
    return res;
}
fp4e2m1_t8 __attribute__((overloadable)) _convert_fp4e2m1_t8(half8 val) {
    fp4e2m1_t8 res;
    res.data.s0 = pack_half_pair_to_fp4(val.s0, val.s1);
    res.data.s1 = pack_half_pair_to_fp4(val.s2, val.s3);
    res.data.s2 = pack_half_pair_to_fp4(val.s4, val.s5);
    res.data.s3 = pack_half_pair_to_fp4(val.s6, val.s7);
    return res;
}
fp4e2m1_t16 __attribute__((overloadable)) _convert_fp4e2m1_t16(half16 val) {
    fp4e2m1_t16 res;
    res.data.s0 = pack_half_pair_to_fp4(val.s0, val.s1);
    res.data.s1 = pack_half_pair_to_fp4(val.s2, val.s3);
    res.data.s2 = pack_half_pair_to_fp4(val.s4, val.s5);
    res.data.s3 = pack_half_pair_to_fp4(val.s6, val.s7);
    res.data.s4 = pack_half_pair_to_fp4(val.s8, val.s9);
    res.data.s5 = pack_half_pair_to_fp4(val.sA, val.sB);
    res.data.s6 = pack_half_pair_to_fp4(val.sC, val.sD);
    res.data.s7 = pack_half_pair_to_fp4(val.sE, val.sF);
    return res;
}

fp4e2m1_t __attribute__((overloadable)) _convert_fp4e2m1_t(float val) {
    fp4e2m1_t res;
    res.data = _intel_convert_f16_to_f4((half)val);
    return res;
}
fp4e2m1_t1 __attribute__((overloadable)) _convert_fp4e2m1_t1(float val[1]) {
    fp4e2m1_t1 res;
    res.data = _intel_convert_f16_to_f4((half)val[0]);
    return res;
}
fp4e2m1_t2 __attribute__((overloadable)) _convert_fp4e2m1_t2(float2 val) {
    fp4e2m1_t2 res;
    res.data = pack_float_pair_to_fp4(val.x, val.y);
    return res;
}
fp4e2m1_t3 __attribute__((overloadable)) _convert_fp4e2m1_t3(float3 val) {
    fp4e2m1_t3 res;
    res.data.s0 = pack_float_pair_to_fp4(val.x, val.y);
    res.data.s1 = _intel_convert_f16_to_f4((half)val.z);
    return res;
}
fp4e2m1_t4 __attribute__((overloadable)) _convert_fp4e2m1_t4(float4 val) {
    fp4e2m1_t4 res;
    res.data.s0 = pack_float_pair_to_fp4(val.x, val.y);
    res.data.s1 = pack_float_pair_to_fp4(val.z, val.w);
    return res;
}
fp4e2m1_t8 __attribute__((overloadable)) _convert_fp4e2m1_t8(float8 val) {
    fp4e2m1_t8 res;
    res.data.s0 = pack_float_pair_to_fp4(val.s0, val.s1);
    res.data.s1 = pack_float_pair_to_fp4(val.s2, val.s3);
    res.data.s2 = pack_float_pair_to_fp4(val.s4, val.s5);
    res.data.s3 = pack_float_pair_to_fp4(val.s6, val.s7);
    return res;
}
fp4e2m1_t16 __attribute__((overloadable)) _convert_fp4e2m1_t16(float16 val) {
    fp4e2m1_t16 res;
    res.data.s0 = pack_float_pair_to_fp4(val.s0, val.s1);
    res.data.s1 = pack_float_pair_to_fp4(val.s2, val.s3);
    res.data.s2 = pack_float_pair_to_fp4(val.s4, val.s5);
    res.data.s3 = pack_float_pair_to_fp4(val.s6, val.s7);
    res.data.s4 = pack_float_pair_to_fp4(val.s8, val.s9);
    res.data.s5 = pack_float_pair_to_fp4(val.sA, val.sB);
    res.data.s6 = pack_float_pair_to_fp4(val.sC, val.sD);
    res.data.s7 = pack_float_pair_to_fp4(val.sE, val.sF);
    return res;
}

fp4e2m1_t __attribute__((overloadable)) _convert_fp4e2m1_t_sat(half val) {
    return _convert_fp4e2m1_t(val);
}
fp4e2m1_t1 __attribute__((overloadable)) _convert_fp4e2m1_t1_sat(half val[1]) {
    return _convert_fp4e2m1_t1(val);
}
fp4e2m1_t2 __attribute__((overloadable)) _convert_fp4e2m1_t2_sat(half2 val) {
    return _convert_fp4e2m1_t2(val);
}
fp4e2m1_t3 __attribute__((overloadable)) _convert_fp4e2m1_t3_sat(half3 val) {
    return _convert_fp4e2m1_t3(val);
}
fp4e2m1_t4 __attribute__((overloadable)) _convert_fp4e2m1_t4_sat(half4 val) {
    return _convert_fp4e2m1_t4(val);
}
fp4e2m1_t8 __attribute__((overloadable)) _convert_fp4e2m1_t8_sat(half8 val) {
    return _convert_fp4e2m1_t8(val);
}
fp4e2m1_t16 __attribute__((overloadable)) _convert_fp4e2m1_t16_sat(half16 val) {
    return _convert_fp4e2m1_t16(val);
}

fp4e2m1_t __attribute__((overloadable)) _convert_fp4e2m1_t_sat(float val) {
    return _convert_fp4e2m1_t(val);
}
fp4e2m1_t1 __attribute__((overloadable)) _convert_fp4e2m1_t1_sat(float val[1]) {
    return _convert_fp4e2m1_t1(val);
}
fp4e2m1_t2 __attribute__((overloadable)) _convert_fp4e2m1_t2_sat(float2 val) {
    return _convert_fp4e2m1_t2(val);
}
fp4e2m1_t3 __attribute__((overloadable)) _convert_fp4e2m1_t3_sat(float3 val) {
    return _convert_fp4e2m1_t3(val);
}
fp4e2m1_t4 __attribute__((overloadable)) _convert_fp4e2m1_t4_sat(float4 val) {
    return _convert_fp4e2m1_t4(val);
}
fp4e2m1_t8 __attribute__((overloadable)) _convert_fp4e2m1_t8_sat(float8 val) {
    return _convert_fp4e2m1_t8(val);
}
fp4e2m1_t16 __attribute__((overloadable)) _convert_fp4e2m1_t16_sat(float16 val) {
    return _convert_fp4e2m1_t16(val);
}

fp4e2m1_t __attribute__((overloadable)) as_fp4e2m1_t(uchar val) {
    fp4e2m1_t res;
    res.data = val;
    return res;
}
fp4e2m1_t1 __attribute__((overloadable)) as_fp4e2m1_t1(uchar val[1]) {
    fp4e2m1_t1 res;
    res.data = val[0];
    return res;
}
fp4e2m1_t2 __attribute__((overloadable)) as_fp4e2m1_t2(uchar2 val) {
    fp4e2m1_t2 res;
    res.data = pack_nibbles(val.x, val.y);
    return res;
}
fp4e2m1_t3 __attribute__((overloadable)) as_fp4e2m1_t3(uchar3 val) {
    fp4e2m1_t3 res;
    res.data.s0 = pack_nibbles(val.x, val.y);
    res.data.s1 = val.z;
    return res;
}
fp4e2m1_t4 __attribute__((overloadable)) as_fp4e2m1_t4(uchar4 val) {
    fp4e2m1_t4 res;
    res.data.s0 = pack_nibbles(val.x, val.y);
    res.data.s1 = pack_nibbles(val.z, val.w);
    return res;
}
fp4e2m1_t8 __attribute__((overloadable)) as_fp4e2m1_t8(uchar8 val) {
    fp4e2m1_t8 res;
    res.data.s0 = pack_nibbles(val.s0, val.s1);
    res.data.s1 = pack_nibbles(val.s2, val.s3);
    res.data.s2 = pack_nibbles(val.s4, val.s5);
    res.data.s3 = pack_nibbles(val.s6, val.s7);
    return res;
}
fp4e2m1_t16 __attribute__((overloadable)) as_fp4e2m1_t16(uchar16 val) {
    fp4e2m1_t16 res;
    res.data.s0 = pack_nibbles(val.s0, val.s1);
    res.data.s1 = pack_nibbles(val.s2, val.s3);
    res.data.s2 = pack_nibbles(val.s4, val.s5);
    res.data.s3 = pack_nibbles(val.s6, val.s7);
    res.data.s4 = pack_nibbles(val.s8, val.s9);
    res.data.s5 = pack_nibbles(val.sA, val.sB);
    res.data.s6 = pack_nibbles(val.sC, val.sD);
    res.data.s7 = pack_nibbles(val.sE, val.sF);
    return res;
}

uchar __attribute__((overloadable)) _as_uchar(fp4e2m1_t val) {
    return val.data;
}
uchar __attribute__((overloadable)) _as_uchar(fp4e2m1_t1 val) {
    return val.data;
}
uchar2 __attribute__((overloadable)) _as_uchar2(fp4e2m1_t2 val) {
    return unpack_nibbles(val.data);
}
uchar3 __attribute__((overloadable)) _as_uchar3(fp4e2m1_t3 val) {
    return (uchar3)(unpack_nibbles(val.data.s0), unpack_nibble_low(val.data.s1));
}
uchar4 __attribute__((overloadable)) _as_uchar4(fp4e2m1_t4 val) {
    return (uchar4)(unpack_nibbles(val.data.s0), 
                    unpack_nibbles(val.data.s1));
}
uchar8 __attribute__((overloadable)) _as_uchar8(fp4e2m1_t8 val) {
    return (uchar8)(unpack_nibbles(val.data.s0), 
                    unpack_nibbles(val.data.s1),
                    unpack_nibbles(val.data.s2), 
                    unpack_nibbles(val.data.s3));
}
uchar16 __attribute__((overloadable)) _as_uchar16(fp4e2m1_t16 val) {
    return (uchar16)(unpack_nibbles(val.data.s0), unpack_nibbles(val.data.s1),
                     unpack_nibbles(val.data.s2), unpack_nibbles(val.data.s3),
                     unpack_nibbles(val.data.s4), unpack_nibbles(val.data.s5),
                     unpack_nibbles(val.data.s6), unpack_nibbles(val.data.s7));
}