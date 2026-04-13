// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// TODO: Replace `_intel_convert*` with bultins when ready, current implementations are copied from XeTLA:

uchar _f16_to_bf8_universal(half val, bool is_saturation) {
    half val_fp16 = val;
    const ushort *p = (ushort *)&val_fp16;
    const ushort mant = p[0] & 0x3FF;
    const uchar exp_mask = 0x7C;

    uchar ret_tmp = p[0] >> 8;
    const bool is_infnan = (ret_tmp & exp_mask) == exp_mask;
    if (is_infnan) {
        ret_tmp |= (mant != 0); // The bit signifying NaNness may have been cut off
    } else {
        ret_tmp += (mant & 0x80) && ((mant & 0x7F) || (mant & 0x100)); // RTE
        if (is_saturation) {
            bool is_overflow = (ret_tmp & exp_mask) == exp_mask;
            ret_tmp -= is_overflow;
        }
    }

    return ret_tmp;
}

uchar _intel_convert_f16_to_bf8(half val) {
    return _f16_to_bf8_universal(val, false);
}

uchar _intel_convert_f16_to_bf8_sat(half val) {
    return _f16_to_bf8_universal(val, true);
}

half _intel_convert_bf8_to_f16(uchar val) {
    ushort temp = val;
    temp = temp << 0x8;
    half temp_fp16 = as_half(temp);
    return temp_fp16;
}

char _f16_to_hf8_universal(half val, bool is_saturation) {
    half val_fp16 = val;
    ushort *p = (ushort *)&val_fp16;
    ushort src = p[0];
    const ushort src_exp_size = 5;
    const ushort src_mant_size = 10;
    const ushort src_exp_bias = (1 << (src_exp_size - 1)) - 1;
    const ushort src_exp_mask = (1 << src_exp_size) - 1;
    const ushort src_mant_mask = (1 << src_mant_size) - 1;
    const short max_exp_unbiased = 8;
    const short min_exp_unbiased = -6;
    const short exp_bias = 7;
    const ushort exp_size = 4;
    const ushort mant_size = 3;
    const uchar nan = 0x7f;
    const uchar max_val = 0x7e;

    ushort src_sign = src >> (src_exp_size + src_mant_size);
    ushort src_exp = (src >> src_mant_size) & src_exp_mask;
    short src_exp_unbiased = src_exp - src_exp_bias;
    ushort src_mant = src & src_mant_mask;

    bool is_src_inf_nan = src_exp == 0x1f;
    bool is_overflow = (src_exp_unbiased > max_exp_unbiased)
            || ((src_exp_unbiased == max_exp_unbiased) && (src_mant > 0x0340));
    bool is_zero = (src_exp_unbiased < (min_exp_unbiased - mant_size));
    bool is_denorm = (src_exp_unbiased < min_exp_unbiased) && (!is_zero);

    uchar dst_val;
    if (is_src_inf_nan) {
        dst_val = nan;
    } else if (is_overflow) {
        dst_val = is_saturation ? max_val : nan;
    } else if (is_zero) {
        dst_val = 0;
    } else if (is_denorm) {
        ushort src_m = src_mant | 0x0400;
        short shift_out_bit = min_exp_unbiased - src_exp_unbiased;
        bool sticky_flag = (src_m & ((1 << shift_out_bit) - 1)) != 0;
        src_m = src_m >> shift_out_bit;
        ushort tail_size = src_mant_size - mant_size;
        sticky_flag
                = sticky_flag || ((src_m & ((1 << (tail_size - 1)) - 1)) != 0);
        bool lsb_bit = src_m & (1 << tail_size);
        bool rnd_bit = src_m & (1 << (tail_size - 1));
        bool carry = (lsb_bit && rnd_bit) || (rnd_bit && sticky_flag);

        dst_val = (src_m >> tail_size) + carry;
    } else {
        ushort tail_size = src_mant_size - mant_size;
        bool sticky_flag = (src_mant & ((1 << (tail_size - 1)) - 1)) != 0;
        bool lsb_bit = src_mant & (1 << tail_size);
        bool rnd_bit = src_mant & (1 << (tail_size - 1));
        bool carry = (lsb_bit && rnd_bit) || (rnd_bit && sticky_flag);
        ushort src_m = (src_mant >> tail_size) + carry;
        ushort src_e = src_exp_unbiased + exp_bias;
        dst_val = (src_e << mant_size) + src_m;
    }
    return src_sign << (exp_size + mant_size) | dst_val;
}

char _intel_convert_f16_to_hf8(half val) {
    return _f16_to_hf8_universal(val, false);
}

char _intel_convert_f16_to_hf8_sat(half val) {
    return _f16_to_hf8_universal(val, true);
}

half _intel_convert_hf8_to_f16(char val) {
    const short exp_bias = 7;
    char data = val;
    ushort sign = data >> 7;
    ushort exp = (data >> 3) & 0b1111;
    ushort mant = data & 0x07;
    ushort dst_val;
    if ((exp == 0xf) && (mant == 0x7)) {
        dst_val = 0x7fff;
    } else if ((exp == 0) && (mant == 0)) {
        dst_val = 0;
    } else if ((exp == 0) && (mant != 0)) {
        ushort lz_count = (mant > 3) ? 0 : ((mant > 1) ? 1 : 2);
        ushort dst_exp = exp - exp_bias + 15 - lz_count;
        ushort dst_mant = (mant << (lz_count + 1)) & 0x7;
        dst_val = (dst_exp << 10) | (dst_mant << 7);
    } else {
        ushort dst_exp = exp - exp_bias + 15;
        dst_val = (dst_exp << 10) | (mant << 7);
    }

    ushort temp = (sign << 15) | dst_val;
    half temp_fp16 = as_half(temp);
    return temp_fp16;
}

uchar _intel_convert_f32_fo_e8m0(float val) {
    uint val_uint = as_uint(val);
    return (uchar)((val_uint >> 23) & 0xFF);
}

uchar _intel_convert_f32_fo_e8m0_sat(float val) {
    return _intel_convert_f32_fo_e8m0(val);
}

float _intel_convert_e8m0_to_f32(uchar val) {
    if (val == 0xFF) {
        return as_float(0xFFC00000); // NaN
    } else if (val == 0) {
        return as_float(0x00400000); // 2^(-127)
    }

    uint temp = (uint)val;
    temp = temp << 23;
    float temp_fp32 = as_float(temp);
    return temp_fp32;
}

typedef struct fp8e5m2_t { uchar data; } fp8e5m2_t;  // bf8
typedef struct fp8e5m2_t1 { uchar data; } fp8e5m2_t1;
typedef struct fp8e5m2_t2 { uchar2 data; } fp8e5m2_t2;
typedef struct fp8e5m2_t3 { uchar3 data; } fp8e5m2_t3;
typedef struct fp8e5m2_t4 { uchar4 data; } fp8e5m2_t4;
typedef struct fp8e5m2_t8 { uchar8 data; } fp8e5m2_t8;
typedef struct fp8e5m2_t16 { uchar16 data; } fp8e5m2_t16;

typedef struct fp8e4m3_t {  char data; } fp8e4m3_t;  // hf8
typedef struct fp8e4m3_t1 { char data; } fp8e4m3_t1;
typedef struct fp8e4m3_t2 { char2 data; } fp8e4m3_t2;
typedef struct fp8e4m3_t3 { char3 data; } fp8e4m3_t3;
typedef struct fp8e4m3_t4 { char4 data; } fp8e4m3_t4;
typedef struct fp8e4m3_t8 { char8 data; } fp8e4m3_t8;
typedef struct fp8e4m3_t16 { char16 data; } fp8e4m3_t16;

typedef struct fp8e8m0_t { uchar data; } fp8e8m0_t;  // e8m0
typedef struct fp8e8m0_t1 { uchar data; } fp8e8m0_t1;
typedef struct fp8e8m0_t2 { uchar2 data; } fp8e8m0_t2;
typedef struct fp8e8m0_t3 { uchar3 data; } fp8e8m0_t3;
typedef struct fp8e8m0_t4 { uchar4 data; } fp8e8m0_t4;
typedef struct fp8e8m0_t8 { uchar8 data; } fp8e8m0_t8;
typedef struct fp8e8m0_t16 { uchar16 data; } fp8e8m0_t16;


half __attribute__((overloadable)) _convert_half(fp8e5m2_t val) {
    return _intel_convert_bf8_to_f16(val.data);
}
half __attribute__((overloadable)) _convert_half(fp8e5m2_t1 val) {
    return _intel_convert_bf8_to_f16(val.data);
}
half2 __attribute__((overloadable)) _convert_half2(fp8e5m2_t2 val) {
    return (half2)(_intel_convert_bf8_to_f16(val.data.s0), _intel_convert_bf8_to_f16(val.data.s1));
}
half3 __attribute__((overloadable)) _convert_half3(fp8e5m2_t3 val) {
    return (half3)(_intel_convert_bf8_to_f16(val.data.s0), _intel_convert_bf8_to_f16(val.data.s1), _intel_convert_bf8_to_f16(val.data.s2));
}
half4 __attribute__((overloadable)) _convert_half4(fp8e5m2_t4 val) {
    return (half4)(_intel_convert_bf8_to_f16(val.data.s0), _intel_convert_bf8_to_f16(val.data.s1),
                  _intel_convert_bf8_to_f16(val.data.s2), _intel_convert_bf8_to_f16(val.data.s3));
}
half8 __attribute__((overloadable)) _convert_half8(fp8e5m2_t8 val) {
    return (half8)(_intel_convert_bf8_to_f16(val.data.s0), _intel_convert_bf8_to_f16(val.data.s1),
                   _intel_convert_bf8_to_f16(val.data.s2), _intel_convert_bf8_to_f16(val.data.s3),
                   _intel_convert_bf8_to_f16(val.data.s4), _intel_convert_bf8_to_f16(val.data.s5),
                   _intel_convert_bf8_to_f16(val.data.s6), _intel_convert_bf8_to_f16(val.data.s7));
}
half16 __attribute__((overloadable)) _convert_half16(fp8e5m2_t16 val) {
    return (half16)(
        _intel_convert_bf8_to_f16(val.data.s0),  _intel_convert_bf8_to_f16(val.data.s1),
        _intel_convert_bf8_to_f16(val.data.s2),  _intel_convert_bf8_to_f16(val.data.s3),
        _intel_convert_bf8_to_f16(val.data.s4),  _intel_convert_bf8_to_f16(val.data.s5),
        _intel_convert_bf8_to_f16(val.data.s6),  _intel_convert_bf8_to_f16(val.data.s7),
        _intel_convert_bf8_to_f16(val.data.s8),  _intel_convert_bf8_to_f16(val.data.s9),
        _intel_convert_bf8_to_f16(val.data.sA), _intel_convert_bf8_to_f16(val.data.sB),
        _intel_convert_bf8_to_f16(val.data.sC), _intel_convert_bf8_to_f16(val.data.sD),
        _intel_convert_bf8_to_f16(val.data.sE), _intel_convert_bf8_to_f16(val.data.sF)
    );
}

float __attribute__((overloadable)) _convert_float(fp8e5m2_t val) {
    return (float)_intel_convert_bf8_to_f16(val.data);
}
float __attribute__((overloadable)) _convert_float(fp8e5m2_t1 val) {
    return (float)_intel_convert_bf8_to_f16(val.data);
}
float2 __attribute__((overloadable)) _convert_float2(fp8e5m2_t2 val) {
    return (float2)(_intel_convert_bf8_to_f16(val.data.s0), _intel_convert_bf8_to_f16(val.data.s1));
}
float3 __attribute__((overloadable)) _convert_float3(fp8e5m2_t3 val) {
    return (float3)(_intel_convert_bf8_to_f16(val.data.s0), _intel_convert_bf8_to_f16(val.data.s1), _intel_convert_bf8_to_f16(val.data.s2));
}
float4 __attribute__((overloadable)) _convert_float4(fp8e5m2_t4 val) {
    return (float4)(_intel_convert_bf8_to_f16(val.data.s0), _intel_convert_bf8_to_f16(val.data.s1),
                   _intel_convert_bf8_to_f16(val.data.s2), _intel_convert_bf8_to_f16(val.data.s3));
}
float8 __attribute__((overloadable)) _convert_float8(fp8e5m2_t8 val) {
    return (float8)(_intel_convert_bf8_to_f16(val.data.s0), _intel_convert_bf8_to_f16(val.data.s1),
                    _intel_convert_bf8_to_f16(val.data.s2), _intel_convert_bf8_to_f16(val.data.s3),
                    _intel_convert_bf8_to_f16(val.data.s4), _intel_convert_bf8_to_f16(val.data.s5),
                    _intel_convert_bf8_to_f16(val.data.s6), _intel_convert_bf8_to_f16(val.data.s7));
}
float16 __attribute__((overloadable)) _convert_float16(fp8e5m2_t16 val) {
    return (float16)(
        _intel_convert_bf8_to_f16(val.data.s0),  _intel_convert_bf8_to_f16(val.data.s1),
        _intel_convert_bf8_to_f16(val.data.s2),  _intel_convert_bf8_to_f16(val.data.s3),
        _intel_convert_bf8_to_f16(val.data.s4),  _intel_convert_bf8_to_f16(val.data.s5),
        _intel_convert_bf8_to_f16(val.data.s6),  _intel_convert_bf8_to_f16(val.data.s7),
        _intel_convert_bf8_to_f16(val.data.s8),  _intel_convert_bf8_to_f16(val.data.s9),
        _intel_convert_bf8_to_f16(val.data.sA), _intel_convert_bf8_to_f16(val.data.sB),
        _intel_convert_bf8_to_f16(val.data.sC), _intel_convert_bf8_to_f16(val.data.sD),
        _intel_convert_bf8_to_f16(val.data.sE), _intel_convert_bf8_to_f16(val.data.sF)
    );
}

half __attribute__((overloadable)) _convert_half(fp8e4m3_t val) {
    return _intel_convert_hf8_to_f16(val.data);
}
half __attribute__((overloadable)) _convert_half(fp8e4m3_t1 val) {
    return _intel_convert_hf8_to_f16(val.data);
}
half2 __attribute__((overloadable)) _convert_half2(fp8e4m3_t2 val) {
    return (half2)(_intel_convert_hf8_to_f16(val.data.s0), _intel_convert_hf8_to_f16(val.data.s1));
}
half3 __attribute__((overloadable)) _convert_half3(fp8e4m3_t3 val) {
    return (half3)(_intel_convert_hf8_to_f16(val.data.s0), _intel_convert_hf8_to_f16(val.data.s1), _intel_convert_hf8_to_f16(val.data.s2));
}
half4 __attribute__((overloadable)) _convert_half4(fp8e4m3_t4 val) {
    return (half4)(_intel_convert_hf8_to_f16(val.data.s0), _intel_convert_hf8_to_f16(val.data.s1),
                  _intel_convert_hf8_to_f16(val.data.s2), _intel_convert_hf8_to_f16(val.data.s3));
}
half8 __attribute__((overloadable)) _convert_half8(fp8e4m3_t8 val) {
    return (half8)(_intel_convert_hf8_to_f16(val.data.s0), _intel_convert_hf8_to_f16(val.data.s1),
                   _intel_convert_hf8_to_f16(val.data.s2), _intel_convert_hf8_to_f16(val.data.s3),
                   _intel_convert_hf8_to_f16(val.data.s4), _intel_convert_hf8_to_f16(val.data.s5),
                   _intel_convert_hf8_to_f16(val.data.s6), _intel_convert_hf8_to_f16(val.data.s7));
}
half16 __attribute__((overloadable)) _convert_half16(fp8e4m3_t16 val) {
    return (half16)(
        _intel_convert_hf8_to_f16(val.data.s0),  _intel_convert_hf8_to_f16(val.data.s1),
        _intel_convert_hf8_to_f16(val.data.s2),  _intel_convert_hf8_to_f16(val.data.s3),
        _intel_convert_hf8_to_f16(val.data.s4),  _intel_convert_hf8_to_f16(val.data.s5),
        _intel_convert_hf8_to_f16(val.data.s6),  _intel_convert_hf8_to_f16(val.data.s7),
        _intel_convert_hf8_to_f16(val.data.s8),  _intel_convert_hf8_to_f16(val.data.s9),
        _intel_convert_hf8_to_f16(val.data.sA), _intel_convert_hf8_to_f16(val.data.sB),
        _intel_convert_hf8_to_f16(val.data.sC), _intel_convert_hf8_to_f16(val.data.sD),
        _intel_convert_hf8_to_f16(val.data.sE), _intel_convert_hf8_to_f16(val.data.sF)
    );
}

float __attribute__((overloadable)) _convert_float(fp8e4m3_t val) {
    return (float)_intel_convert_hf8_to_f16(val.data);
}
float __attribute__((overloadable)) _convert_float(fp8e4m3_t1 val) {
    return (float)_intel_convert_hf8_to_f16(val.data);
}
float2 __attribute__((overloadable)) _convert_float2(fp8e4m3_t2 val) {
    return (float2)(_intel_convert_hf8_to_f16(val.data.s0), _intel_convert_hf8_to_f16(val.data.s1));
}
float3 __attribute__((overloadable)) _convert_float3(fp8e4m3_t3 val) {
    return (float3)(_intel_convert_hf8_to_f16(val.data.s0), _intel_convert_hf8_to_f16(val.data.s1), _intel_convert_hf8_to_f16(val.data.s2));
}
float4 __attribute__((overloadable)) _convert_float4(fp8e4m3_t4 val) {
    return (float4)(_intel_convert_hf8_to_f16(val.data.s0), _intel_convert_hf8_to_f16(val.data.s1),
                   _intel_convert_hf8_to_f16(val.data.s2), _intel_convert_hf8_to_f16(val.data.s3));
}
float8 __attribute__((overloadable)) _convert_float8(fp8e4m3_t8 val) {
    return (float8)(_intel_convert_hf8_to_f16(val.data.s0), _intel_convert_hf8_to_f16(val.data.s1),
                    _intel_convert_hf8_to_f16(val.data.s2), _intel_convert_hf8_to_f16(val.data.s3),
                    _intel_convert_hf8_to_f16(val.data.s4), _intel_convert_hf8_to_f16(val.data.s5),
                    _intel_convert_hf8_to_f16(val.data.s6), _intel_convert_hf8_to_f16(val.data.s7));
}
float16 __attribute__((overloadable)) _convert_float16(fp8e4m3_t16 val) {
    return (float16)(
        _intel_convert_hf8_to_f16(val.data.s0),  _intel_convert_hf8_to_f16(val.data.s1),
        _intel_convert_hf8_to_f16(val.data.s2),  _intel_convert_hf8_to_f16(val.data.s3),
        _intel_convert_hf8_to_f16(val.data.s4),  _intel_convert_hf8_to_f16(val.data.s5),
        _intel_convert_hf8_to_f16(val.data.s6),  _intel_convert_hf8_to_f16(val.data.s7),
        _intel_convert_hf8_to_f16(val.data.s8),  _intel_convert_hf8_to_f16(val.data.s9),
        _intel_convert_hf8_to_f16(val.data.sA), _intel_convert_hf8_to_f16(val.data.sB),
        _intel_convert_hf8_to_f16(val.data.sC), _intel_convert_hf8_to_f16(val.data.sD),
        _intel_convert_hf8_to_f16(val.data.sE), _intel_convert_hf8_to_f16(val.data.sF)
    );
}

float __attribute__((overloadable)) _convert_float(fp8e8m0_t val) {
    return (float)_intel_convert_e8m0_to_f32(val.data);
}
float __attribute__((overloadable)) _convert_float(fp8e8m0_t1 val) {
    return (float)_intel_convert_e8m0_to_f32(val.data);
}
float2 __attribute__((overloadable)) _convert_float2(fp8e8m0_t2 val) {
    return (float2)(_intel_convert_e8m0_to_f32(val.data.s0), _intel_convert_e8m0_to_f32(val.data.s1));
}
float3 __attribute__((overloadable)) _convert_float3(fp8e8m0_t3 val) {
    return (float3)(_intel_convert_e8m0_to_f32(val.data.s0), _intel_convert_e8m0_to_f32(val.data.s1), _intel_convert_e8m0_to_f32(val.data.s2));
}
float4 __attribute__((overloadable)) _convert_float4(fp8e8m0_t4 val) {
    return (float4)(_intel_convert_e8m0_to_f32(val.data.s0), _intel_convert_e8m0_to_f32(val.data.s1),
                   _intel_convert_e8m0_to_f32(val.data.s2), _intel_convert_e8m0_to_f32(val.data.s3));
}
float8 __attribute__((overloadable)) _convert_float8(fp8e8m0_t8 val) {
    return (float8)(_intel_convert_e8m0_to_f32(val.data.s0), _intel_convert_e8m0_to_f32(val.data.s1),
                    _intel_convert_e8m0_to_f32(val.data.s2), _intel_convert_e8m0_to_f32(val.data.s3),
                    _intel_convert_e8m0_to_f32(val.data.s4), _intel_convert_e8m0_to_f32(val.data.s5),
                    _intel_convert_e8m0_to_f32(val.data.s6), _intel_convert_e8m0_to_f32(val.data.s7));
}
float16 __attribute__((overloadable)) _convert_float16(fp8e8m0_t16 val) {
    return (float16)(
        _intel_convert_e8m0_to_f32(val.data.s0),  _intel_convert_e8m0_to_f32(val.data.s1),
        _intel_convert_e8m0_to_f32(val.data.s2),  _intel_convert_e8m0_to_f32(val.data.s3),
        _intel_convert_e8m0_to_f32(val.data.s4),  _intel_convert_e8m0_to_f32(val.data.s5),
        _intel_convert_e8m0_to_f32(val.data.s6),  _intel_convert_e8m0_to_f32(val.data.s7),
        _intel_convert_e8m0_to_f32(val.data.s8),  _intel_convert_e8m0_to_f32(val.data.s9),
        _intel_convert_e8m0_to_f32(val.data.sA), _intel_convert_e8m0_to_f32(val.data.sB),
        _intel_convert_e8m0_to_f32(val.data.sC), _intel_convert_e8m0_to_f32(val.data.sD),
        _intel_convert_e8m0_to_f32(val.data.sE), _intel_convert_e8m0_to_f32(val.data.sF)
    );
}

fp8e5m2_t __attribute__((overloadable)) _convert_fp8e5m2_t(float val) {
    fp8e5m2_t res;
    res.data = _intel_convert_f16_to_bf8((half)val);
    return res;
}
fp8e5m2_t1 __attribute__((overloadable)) _convert_fp8e5m2_t1(float val[1]) {
    fp8e5m2_t1 res;
    res.data = _intel_convert_f16_to_bf8((half)val[0]);
    return res;
}
fp8e5m2_t2 __attribute__((overloadable)) _convert_fp8e5m2_t2(float2 val) {
    fp8e5m2_t2 res;
    res.data.s0 = _intel_convert_f16_to_bf8((half)val.x);
    res.data.s1 = _intel_convert_f16_to_bf8((half)val.y);
    return res;
}
fp8e5m2_t3 __attribute__((overloadable)) _convert_fp8e5m2_t3(float3 val) {
    fp8e5m2_t3 res;
    res.data.s0 = _intel_convert_f16_to_bf8((half)val.x);
    res.data.s1 = _intel_convert_f16_to_bf8((half)val.y);
    res.data.s2 = _intel_convert_f16_to_bf8((half)val.z);
    return res;
}
fp8e5m2_t4 __attribute__((overloadable)) _convert_fp8e5m2_t4(float4 val) {
    fp8e5m2_t4 res;
    res.data.s0 = _intel_convert_f16_to_bf8((half)val.x);
    res.data.s1 = _intel_convert_f16_to_bf8((half)val.y);
    res.data.s2 = _intel_convert_f16_to_bf8((half)val.z);
    res.data.s3 = _intel_convert_f16_to_bf8((half)val.w);
    return res;
}
fp8e5m2_t8 __attribute__((overloadable)) _convert_fp8e5m2_t8(float8 val) {
    fp8e5m2_t8 res;
    res.data.s0 = _intel_convert_f16_to_bf8((half)val.s0);
    res.data.s1 = _intel_convert_f16_to_bf8((half)val.s1);
    res.data.s2 = _intel_convert_f16_to_bf8((half)val.s2);
    res.data.s3 = _intel_convert_f16_to_bf8((half)val.s3);
    res.data.s4 = _intel_convert_f16_to_bf8((half)val.s4);
    res.data.s5 = _intel_convert_f16_to_bf8((half)val.s5);
    res.data.s6 = _intel_convert_f16_to_bf8((half)val.s6);
    res.data.s7 = _intel_convert_f16_to_bf8((half)val.s7);
    return res;
}
fp8e5m2_t16 __attribute__((overloadable)) _convert_fp8e5m2_t16(float16 val) {
    fp8e5m2_t16 res;
    res.data.s0  = _intel_convert_f16_to_bf8((half)val.s0);
    res.data.s1  = _intel_convert_f16_to_bf8((half)val.s1);
    res.data.s2  = _intel_convert_f16_to_bf8((half)val.s2);
    res.data.s3  = _intel_convert_f16_to_bf8((half)val.s3);
    res.data.s4  = _intel_convert_f16_to_bf8((half)val.s4);
    res.data.s5  = _intel_convert_f16_to_bf8((half)val.s5);
    res.data.s6  = _intel_convert_f16_to_bf8((half)val.s6);
    res.data.s7  = _intel_convert_f16_to_bf8((half)val.s7);
    res.data.s8  = _intel_convert_f16_to_bf8((half)val.s8);
    res.data.s9  = _intel_convert_f16_to_bf8((half)val.s9);
    res.data.sA = _intel_convert_f16_to_bf8((half)val.sA);
    res.data.sB = _intel_convert_f16_to_bf8((half)val.sB);
    res.data.sC = _intel_convert_f16_to_bf8((half)val.sC);
    res.data.sD = _intel_convert_f16_to_bf8((half)val.sD);
    res.data.sE = _intel_convert_f16_to_bf8((half)val.sE);
    res.data.sF = _intel_convert_f16_to_bf8((half)val.sF);
    return res;
}

fp8e5m2_t __attribute__((overloadable)) _convert_fp8e5m2_t(half val) {
    fp8e5m2_t res;
    res.data = _intel_convert_f16_to_bf8(val);
    return res;
}
fp8e5m2_t1 __attribute__((overloadable)) _convert_fp8e5m2_t1(half val[1]) {
    fp8e5m2_t1 res;
    res.data = _intel_convert_f16_to_bf8(val[0]);
    return res;
}
fp8e5m2_t2 __attribute__((overloadable)) _convert_fp8e5m2_t2(half2 val) {
    fp8e5m2_t2 res;
    res.data.s0 = _intel_convert_f16_to_bf8(val.x);
    res.data.s1 = _intel_convert_f16_to_bf8(val.y);
    return res;
}
fp8e5m2_t3 __attribute__((overloadable)) _convert_fp8e5m2_t3(half3 val) {
    fp8e5m2_t3 res;
    res.data.s0 = _intel_convert_f16_to_bf8(val.x);
    res.data.s1 = _intel_convert_f16_to_bf8(val.y);
    res.data.s2 = _intel_convert_f16_to_bf8(val.z);
    return res;
}
fp8e5m2_t4 __attribute__((overloadable)) _convert_fp8e5m2_t4(half4 val) {
    fp8e5m2_t4 res;
    res.data.s0 = _intel_convert_f16_to_bf8(val.x);
    res.data.s1 = _intel_convert_f16_to_bf8(val.y);
    res.data.s2 = _intel_convert_f16_to_bf8(val.z);
    res.data.s3 = _intel_convert_f16_to_bf8(val.w);
    return res;
}
fp8e5m2_t8 __attribute__((overloadable)) _convert_fp8e5m2_t8(half8 val) {
    fp8e5m2_t8 res;
    res.data.s0 = _intel_convert_f16_to_bf8(val.s0);
    res.data.s1 = _intel_convert_f16_to_bf8(val.s1);
    res.data.s2 = _intel_convert_f16_to_bf8(val.s2);
    res.data.s3 = _intel_convert_f16_to_bf8(val.s3);
    res.data.s4 = _intel_convert_f16_to_bf8(val.s4);
    res.data.s5 = _intel_convert_f16_to_bf8(val.s5);
    res.data.s6 = _intel_convert_f16_to_bf8(val.s6);
    res.data.s7 = _intel_convert_f16_to_bf8(val.s7);
    return res;
}
fp8e5m2_t16 __attribute__((overloadable)) _convert_fp8e5m2_t16(half16 val) {
    fp8e5m2_t16 res;
    res.data.s0  = _intel_convert_f16_to_bf8(val.s0);
    res.data.s1  = _intel_convert_f16_to_bf8(val.s1);
    res.data.s2  = _intel_convert_f16_to_bf8(val.s2);
    res.data.s3  = _intel_convert_f16_to_bf8(val.s3);
    res.data.s4  = _intel_convert_f16_to_bf8(val.s4);
    res.data.s5  = _intel_convert_f16_to_bf8(val.s5);
    res.data.s6  = _intel_convert_f16_to_bf8(val.s6);
    res.data.s7  = _intel_convert_f16_to_bf8(val.s7);
    res.data.s8  = _intel_convert_f16_to_bf8(val.s8);
    res.data.s9  = _intel_convert_f16_to_bf8(val.s9);
    res.data.sA = _intel_convert_f16_to_bf8(val.sA);
    res.data.sB = _intel_convert_f16_to_bf8(val.sB);
    res.data.sC = _intel_convert_f16_to_bf8(val.sC);
    res.data.sD = _intel_convert_f16_to_bf8(val.sD);
    res.data.sE = _intel_convert_f16_to_bf8(val.sE);
    res.data.sF = _intel_convert_f16_to_bf8(val.sF);
    return res;
}

fp8e5m2_t __attribute__((overloadable)) _convert_fp8e5m2_t_sat(float val) {
    fp8e5m2_t res;
    res.data = _intel_convert_f16_to_bf8_sat((half)val);
    return res;
}
fp8e5m2_t1 __attribute__((overloadable)) _convert_fp8e5m2_t1_sat(float val[1]) {
    fp8e5m2_t1 res;
    res.data = _intel_convert_f16_to_bf8_sat((half)val[0]);
    return res;
}
fp8e5m2_t2 __attribute__((overloadable)) _convert_fp8e5m2_t2_sat(float2 val) {
    fp8e5m2_t2 res;
    res.data.s0 = _intel_convert_f16_to_bf8_sat((half)val.x);
    res.data.s1 = _intel_convert_f16_to_bf8_sat((half)val.y);
    return res;
}
fp8e5m2_t3 __attribute__((overloadable)) _convert_fp8e5m2_t3_sat(float3 val) {
    fp8e5m2_t3 res;
    res.data.s0 = _intel_convert_f16_to_bf8_sat((half)val.x);
    res.data.s1 = _intel_convert_f16_to_bf8_sat((half)val.y);
    res.data.s2 = _intel_convert_f16_to_bf8_sat((half)val.z);
    return res;
}
fp8e5m2_t4 __attribute__((overloadable)) _convert_fp8e5m2_t4_sat(float4 val) {
    fp8e5m2_t4 res;
    res.data.s0 = _intel_convert_f16_to_bf8_sat((half)val.x);
    res.data.s1 = _intel_convert_f16_to_bf8_sat((half)val.y);
    res.data.s2 = _intel_convert_f16_to_bf8_sat((half)val.z);
    res.data.s3 = _intel_convert_f16_to_bf8_sat((half)val.w);
    return res;
}
fp8e5m2_t8 __attribute__((overloadable)) _convert_fp8e5m2_t8_sat(float8 val) {
    fp8e5m2_t8 res;
    res.data.s0 = _intel_convert_f16_to_bf8_sat((half)val.s0);
    res.data.s1 = _intel_convert_f16_to_bf8_sat((half)val.s1);
    res.data.s2 = _intel_convert_f16_to_bf8_sat((half)val.s2);
    res.data.s3 = _intel_convert_f16_to_bf8_sat((half)val.s3);
    res.data.s4 = _intel_convert_f16_to_bf8_sat((half)val.s4);
    res.data.s5 = _intel_convert_f16_to_bf8_sat((half)val.s5);
    res.data.s6 = _intel_convert_f16_to_bf8_sat((half)val.s6);
    res.data.s7 = _intel_convert_f16_to_bf8_sat((half)val.s7);
    return res;
}
fp8e5m2_t16 __attribute__((overloadable)) _convert_fp8e5m2_t16_sat(float16 val) {
    fp8e5m2_t16 res;
    res.data.s0  = _intel_convert_f16_to_bf8_sat((half)val.s0);
    res.data.s1  = _intel_convert_f16_to_bf8_sat((half)val.s1);
    res.data.s2  = _intel_convert_f16_to_bf8_sat((half)val.s2);
    res.data.s3  = _intel_convert_f16_to_bf8_sat((half)val.s3);
    res.data.s4  = _intel_convert_f16_to_bf8_sat((half)val.s4);
    res.data.s5  = _intel_convert_f16_to_bf8_sat((half)val.s5);
    res.data.s6  = _intel_convert_f16_to_bf8_sat((half)val.s6);
    res.data.s7  = _intel_convert_f16_to_bf8_sat((half)val.s7);
    res.data.s8  = _intel_convert_f16_to_bf8_sat((half)val.s8);
    res.data.s9  = _intel_convert_f16_to_bf8_sat((half)val.s9);
    res.data.sA = _intel_convert_f16_to_bf8_sat((half)val.sA);
    res.data.sB = _intel_convert_f16_to_bf8_sat((half)val.sB);
    res.data.sC = _intel_convert_f16_to_bf8_sat((half)val.sC);
    res.data.sD = _intel_convert_f16_to_bf8_sat((half)val.sD);
    res.data.sE = _intel_convert_f16_to_bf8_sat((half)val.sE);
    res.data.sF = _intel_convert_f16_to_bf8_sat((half)val.sF);
    return res;
}

fp8e5m2_t __attribute__((overloadable)) _convert_fp8e5m2_t_sat(half val) {
    fp8e5m2_t res;
    res.data = _intel_convert_f16_to_bf8_sat(val);
    return res;
}
fp8e5m2_t1 __attribute__((overloadable)) _convert_fp8e5m2_t1_sat(half val[1]) {
    fp8e5m2_t1 res;
    res.data = _intel_convert_f16_to_bf8_sat(val[0]);
    return res;
}
fp8e5m2_t2 __attribute__((overloadable)) _convert_fp8e5m2_t2_sat(half2 val) {
    fp8e5m2_t2 res;
    res.data.s0 = _intel_convert_f16_to_bf8_sat(val.x);
    res.data.s1 = _intel_convert_f16_to_bf8_sat(val.y);
    return res;
}
fp8e5m2_t3 __attribute__((overloadable)) _convert_fp8e5m2_t3_sat(half3 val) {
    fp8e5m2_t3 res;
    res.data.s0 = _intel_convert_f16_to_bf8_sat(val.x);
    res.data.s1 = _intel_convert_f16_to_bf8_sat(val.y);
    res.data.s2 = _intel_convert_f16_to_bf8_sat(val.z);
    return res;
}
fp8e5m2_t4 __attribute__((overloadable)) _convert_fp8e5m2_t4_sat(half4 val) {
    fp8e5m2_t4 res;
    res.data.s0 = _intel_convert_f16_to_bf8_sat(val.x);
    res.data.s1 = _intel_convert_f16_to_bf8_sat(val.y);
    res.data.s2 = _intel_convert_f16_to_bf8_sat(val.z);
    res.data.s3 = _intel_convert_f16_to_bf8_sat(val.w);
    return res;
}
fp8e5m2_t8 __attribute__((overloadable)) _convert_fp8e5m2_t8_sat(half8 val) {
    fp8e5m2_t8 res;
    res.data.s0 = _intel_convert_f16_to_bf8_sat(val.s0);
    res.data.s1 = _intel_convert_f16_to_bf8_sat(val.s1);
    res.data.s2 = _intel_convert_f16_to_bf8_sat(val.s2);
    res.data.s3 = _intel_convert_f16_to_bf8_sat(val.s3);
    res.data.s4 = _intel_convert_f16_to_bf8_sat(val.s4);
    res.data.s5 = _intel_convert_f16_to_bf8_sat(val.s5);
    res.data.s6 = _intel_convert_f16_to_bf8_sat(val.s6);
    res.data.s7 = _intel_convert_f16_to_bf8_sat(val.s7);
    return res;
}
fp8e5m2_t16 __attribute__((overloadable)) _convert_fp8e5m2_t16_sat(half16 val) {
    fp8e5m2_t16 res;
    res.data.s0  = _intel_convert_f16_to_bf8_sat(val.s0);
    res.data.s1  = _intel_convert_f16_to_bf8_sat(val.s1);
    res.data.s2  = _intel_convert_f16_to_bf8_sat(val.s2);
    res.data.s3  = _intel_convert_f16_to_bf8_sat(val.s3);
    res.data.s4  = _intel_convert_f16_to_bf8_sat(val.s4);
    res.data.s5  = _intel_convert_f16_to_bf8_sat(val.s5);
    res.data.s6  = _intel_convert_f16_to_bf8_sat(val.s6);
    res.data.s7  = _intel_convert_f16_to_bf8_sat(val.s7);
    res.data.s8  = _intel_convert_f16_to_bf8_sat(val.s8);
    res.data.s9  = _intel_convert_f16_to_bf8_sat(val.s9);
    res.data.sA = _intel_convert_f16_to_bf8_sat(val.sA);
    res.data.sB = _intel_convert_f16_to_bf8_sat(val.sB);
    res.data.sC = _intel_convert_f16_to_bf8_sat(val.sC);
    res.data.sD = _intel_convert_f16_to_bf8_sat(val.sD);
    res.data.sE = _intel_convert_f16_to_bf8_sat(val.sE);
    res.data.sF = _intel_convert_f16_to_bf8_sat(val.sF);
    return res;
}

fp8e4m3_t __attribute__((overloadable)) _convert_fp8e4m3_t(float val) {
    fp8e4m3_t res;
    res.data = _intel_convert_f16_to_hf8((half)val);
    return res;
}
fp8e4m3_t1 __attribute__((overloadable)) _convert_fp8e4m3_t1(float val[1]) {
    fp8e4m3_t1 res;
    res.data = _intel_convert_f16_to_hf8((half)val[0]);
    return res;
}
fp8e4m3_t2 __attribute__((overloadable)) _convert_fp8e4m3_t2(float2 val) {
    fp8e4m3_t2 res;
    res.data.s0 = _intel_convert_f16_to_hf8((half)val.x);
    res.data.s1 = _intel_convert_f16_to_hf8((half)val.y);
    return res;
}
fp8e4m3_t3 __attribute__((overloadable)) _convert_fp8e4m3_t3(float3 val) {
    fp8e4m3_t3 res;
    res.data.s0 = _intel_convert_f16_to_hf8((half)val.x);
    res.data.s1 = _intel_convert_f16_to_hf8((half)val.y);
    res.data.s2 = _intel_convert_f16_to_hf8((half)val.z);
    return res;
}
fp8e4m3_t4 __attribute__((overloadable)) _convert_fp8e4m3_t4(float4 val) {
    fp8e4m3_t4 res;
    res.data.s0 = _intel_convert_f16_to_hf8((half)val.x);
    res.data.s1 = _intel_convert_f16_to_hf8((half)val.y);
    res.data.s2 = _intel_convert_f16_to_hf8((half)val.z);
    res.data.s3 = _intel_convert_f16_to_hf8((half)val.w);
    return res;
}
fp8e4m3_t8 __attribute__((overloadable)) _convert_fp8e4m3_t8(float8 val) {
    fp8e4m3_t8 res;
    res.data.s0 = _intel_convert_f16_to_hf8((half)val.s0);
    res.data.s1 = _intel_convert_f16_to_hf8((half)val.s1);
    res.data.s2 = _intel_convert_f16_to_hf8((half)val.s2);
    res.data.s3 = _intel_convert_f16_to_hf8((half)val.s3);
    res.data.s4 = _intel_convert_f16_to_hf8((half)val.s4);
    res.data.s5 = _intel_convert_f16_to_hf8((half)val.s5);
    res.data.s6 = _intel_convert_f16_to_hf8((half)val.s6);
    res.data.s7 = _intel_convert_f16_to_hf8((half)val.s7);
    return res;
}
fp8e4m3_t16 __attribute__((overloadable)) _convert_fp8e4m3_t16(float16 val) {
    fp8e4m3_t16 res;
    res.data.s0  = _intel_convert_f16_to_hf8((half)val.s0);
    res.data.s1  = _intel_convert_f16_to_hf8((half)val.s1);
    res.data.s2  = _intel_convert_f16_to_hf8((half)val.s2);
    res.data.s3  = _intel_convert_f16_to_hf8((half)val.s3);
    res.data.s4  = _intel_convert_f16_to_hf8((half)val.s4);
    res.data.s5  = _intel_convert_f16_to_hf8((half)val.s5);
    res.data.s6  = _intel_convert_f16_to_hf8((half)val.s6);
    res.data.s7  = _intel_convert_f16_to_hf8((half)val.s7);
    res.data.s8  = _intel_convert_f16_to_hf8((half)val.s8);
    res.data.s9  = _intel_convert_f16_to_hf8((half)val.s9);
    res.data.sA = _intel_convert_f16_to_hf8((half)val.sA);
    res.data.sB = _intel_convert_f16_to_hf8((half)val.sB);
    res.data.sC = _intel_convert_f16_to_hf8((half)val.sC);
    res.data.sD = _intel_convert_f16_to_hf8((half)val.sD);
    res.data.sE = _intel_convert_f16_to_hf8((half)val.sE);
    res.data.sF = _intel_convert_f16_to_hf8((half)val.sF);
    return res;
}

fp8e4m3_t __attribute__((overloadable)) _convert_fp8e4m3_t(half val) {
    fp8e4m3_t res;
    res.data = _intel_convert_f16_to_hf8(val);
    return res;
}
fp8e4m3_t1 __attribute__((overloadable)) _convert_fp8e4m3_t1(half val[1]) {
    fp8e4m3_t1 res;
    res.data = _intel_convert_f16_to_hf8(val[0]);
    return res;
}
fp8e4m3_t2 __attribute__((overloadable)) _convert_fp8e4m3_t2(half2 val) {
    fp8e4m3_t2 res;
    res.data.s0 = _intel_convert_f16_to_hf8(val.x);
    res.data.s1 = _intel_convert_f16_to_hf8(val.y);
    return res;
}
fp8e4m3_t3 __attribute__((overloadable)) _convert_fp8e4m3_t3(half3 val) {
    fp8e4m3_t3 res;
    res.data.s0 = _intel_convert_f16_to_hf8(val.x);
    res.data.s1 = _intel_convert_f16_to_hf8(val.y);
    res.data.s2 = _intel_convert_f16_to_hf8(val.z);
    return res;
}
fp8e4m3_t4 __attribute__((overloadable)) _convert_fp8e4m3_t4(half4 val) {
    fp8e4m3_t4 res;
    res.data.s0 = _intel_convert_f16_to_hf8(val.x);
    res.data.s1 = _intel_convert_f16_to_hf8(val.y);
    res.data.s2 = _intel_convert_f16_to_hf8(val.z);
    res.data.s3 = _intel_convert_f16_to_hf8(val.w);
    return res;
}
fp8e4m3_t8 __attribute__((overloadable)) _convert_fp8e4m3_t8(half8 val) {
    fp8e4m3_t8 res;
    res.data.s0 = _intel_convert_f16_to_hf8(val.s0);
    res.data.s1 = _intel_convert_f16_to_hf8(val.s1);
    res.data.s2 = _intel_convert_f16_to_hf8(val.s2);
    res.data.s3 = _intel_convert_f16_to_hf8(val.s3);
    res.data.s4 = _intel_convert_f16_to_hf8(val.s4);
    res.data.s5 = _intel_convert_f16_to_hf8(val.s5);
    res.data.s6 = _intel_convert_f16_to_hf8(val.s6);
    res.data.s7 = _intel_convert_f16_to_hf8(val.s7);
    return res;
}
fp8e4m3_t16 __attribute__((overloadable)) _convert_fp8e4m3_t16(half16 val) {
    fp8e4m3_t16 res;
    res.data.s0  = _intel_convert_f16_to_hf8(val.s0);
    res.data.s1  = _intel_convert_f16_to_hf8(val.s1);
    res.data.s2  = _intel_convert_f16_to_hf8(val.s2);
    res.data.s3  = _intel_convert_f16_to_hf8(val.s3);
    res.data.s4  = _intel_convert_f16_to_hf8(val.s4);
    res.data.s5  = _intel_convert_f16_to_hf8(val.s5);
    res.data.s6  = _intel_convert_f16_to_hf8(val.s6);
    res.data.s7  = _intel_convert_f16_to_hf8(val.s7);
    res.data.s8  = _intel_convert_f16_to_hf8(val.s8);
    res.data.s9  = _intel_convert_f16_to_hf8(val.s9);
    res.data.sA = _intel_convert_f16_to_hf8(val.sA);
    res.data.sB = _intel_convert_f16_to_hf8(val.sB);
    res.data.sC = _intel_convert_f16_to_hf8(val.sC);
    res.data.sD = _intel_convert_f16_to_hf8(val.sD);
    res.data.sE = _intel_convert_f16_to_hf8(val.sE);
    res.data.sF = _intel_convert_f16_to_hf8(val.sF);
    return res;
}

fp8e4m3_t __attribute__((overloadable)) _convert_fp8e4m3_t_sat(float val) {
    fp8e4m3_t res;
    res.data = _intel_convert_f16_to_hf8_sat((half)val);
    return res;
}
fp8e4m3_t1 __attribute__((overloadable)) _convert_fp8e4m3_t1_sat(float val[1]) {
    fp8e4m3_t1 res;
    res.data = _intel_convert_f16_to_hf8_sat((half)val[0]);
    return res;
}
fp8e4m3_t2 __attribute__((overloadable)) _convert_fp8e4m3_t2_sat(float2 val) {
    fp8e4m3_t2 res;
    res.data.s0 = _intel_convert_f16_to_hf8_sat((half)val.x);
    res.data.s1 = _intel_convert_f16_to_hf8_sat((half)val.y);
    return res;
}
fp8e4m3_t3 __attribute__((overloadable)) _convert_fp8e4m3_t3_sat(float3 val) {
    fp8e4m3_t3 res;
    res.data.s0 = _intel_convert_f16_to_hf8_sat((half)val.x);
    res.data.s1 = _intel_convert_f16_to_hf8_sat((half)val.y);
    res.data.s2 = _intel_convert_f16_to_hf8_sat((half)val.z);
    return res;
}
fp8e4m3_t4 __attribute__((overloadable)) _convert_fp8e4m3_t4_sat(float4 val) {
    fp8e4m3_t4 res;
    res.data.s0 = _intel_convert_f16_to_hf8_sat((half)val.x);
    res.data.s1 = _intel_convert_f16_to_hf8_sat((half)val.y);
    res.data.s2 = _intel_convert_f16_to_hf8_sat((half)val.z);
    res.data.s3 = _intel_convert_f16_to_hf8_sat((half)val.w);
    return res;
}
fp8e4m3_t8 __attribute__((overloadable)) _convert_fp8e4m3_t8_sat(float8 val) {
    fp8e4m3_t8 res;
    res.data.s0 = _intel_convert_f16_to_hf8_sat((half)val.s0);
    res.data.s1 = _intel_convert_f16_to_hf8_sat((half)val.s1);
    res.data.s2 = _intel_convert_f16_to_hf8_sat((half)val.s2);
    res.data.s3 = _intel_convert_f16_to_hf8_sat((half)val.s3);
    res.data.s4 = _intel_convert_f16_to_hf8_sat((half)val.s4);
    res.data.s5 = _intel_convert_f16_to_hf8_sat((half)val.s5);
    res.data.s6 = _intel_convert_f16_to_hf8_sat((half)val.s6);
    res.data.s7 = _intel_convert_f16_to_hf8_sat((half)val.s7);
    return res;
}
fp8e4m3_t16 __attribute__((overloadable)) _convert_fp8e4m3_t16_sat(float16 val) {
    fp8e4m3_t16 res;
    res.data.s0  = _intel_convert_f16_to_hf8_sat((half)val.s0);
    res.data.s1  = _intel_convert_f16_to_hf8_sat((half)val.s1);
    res.data.s2  = _intel_convert_f16_to_hf8_sat((half)val.s2);
    res.data.s3  = _intel_convert_f16_to_hf8_sat((half)val.s3);
    res.data.s4  = _intel_convert_f16_to_hf8_sat((half)val.s4);
    res.data.s5  = _intel_convert_f16_to_hf8_sat((half)val.s5);
    res.data.s6  = _intel_convert_f16_to_hf8_sat((half)val.s6);
    res.data.s7  = _intel_convert_f16_to_hf8_sat((half)val.s7);
    res.data.s8  = _intel_convert_f16_to_hf8_sat((half)val.s8);
    res.data.s9  = _intel_convert_f16_to_hf8_sat((half)val.s9);
    res.data.sA = _intel_convert_f16_to_hf8_sat((half)val.sA);
    res.data.sB = _intel_convert_f16_to_hf8_sat((half)val.sB);
    res.data.sC = _intel_convert_f16_to_hf8_sat((half)val.sC);
    res.data.sD = _intel_convert_f16_to_hf8_sat((half)val.sD);
    res.data.sE = _intel_convert_f16_to_hf8_sat((half)val.sE);
    res.data.sF = _intel_convert_f16_to_hf8_sat((half)val.sF);
    return res;
}

fp8e4m3_t __attribute__((overloadable)) _convert_fp8e4m3_t_sat(half val) {
    fp8e4m3_t res;
    res.data = _intel_convert_f16_to_hf8_sat(val);
    return res;
}
fp8e4m3_t1 __attribute__((overloadable)) _convert_fp8e4m3_t1_sat(half val[1]) {
    fp8e4m3_t1 res;
    res.data = _intel_convert_f16_to_hf8_sat(val[0]);
    return res;
}
fp8e4m3_t2 __attribute__((overloadable)) _convert_fp8e4m3_t2_sat(half2 val) {
    fp8e4m3_t2 res;
    res.data.s0 = _intel_convert_f16_to_hf8_sat(val.x);
    res.data.s1 = _intel_convert_f16_to_hf8_sat(val.y);
    return res;
}
fp8e4m3_t3 __attribute__((overloadable)) _convert_fp8e4m3_t3_sat(half3 val) {
    fp8e4m3_t3 res;
    res.data.s0 = _intel_convert_f16_to_hf8_sat(val.x);
    res.data.s1 = _intel_convert_f16_to_hf8_sat(val.y);
    res.data.s2 = _intel_convert_f16_to_hf8_sat(val.z);
    return res;
}
fp8e4m3_t4 __attribute__((overloadable)) _convert_fp8e4m3_t4_sat(half4 val) {
    fp8e4m3_t4 res;
    res.data.s0 = _intel_convert_f16_to_hf8_sat(val.x);
    res.data.s1 = _intel_convert_f16_to_hf8_sat(val.y);
    res.data.s2 = _intel_convert_f16_to_hf8_sat(val.z);
    res.data.s3 = _intel_convert_f16_to_hf8_sat(val.w);
    return res;
}
fp8e4m3_t8 __attribute__((overloadable)) _convert_fp8e4m3_t8_sat(half8 val) {
    fp8e4m3_t8 res;
    res.data.s0 = _intel_convert_f16_to_hf8_sat(val.s0);
    res.data.s1 = _intel_convert_f16_to_hf8_sat(val.s1);
    res.data.s2 = _intel_convert_f16_to_hf8_sat(val.s2);
    res.data.s3 = _intel_convert_f16_to_hf8_sat(val.s3);
    res.data.s4 = _intel_convert_f16_to_hf8_sat(val.s4);
    res.data.s5 = _intel_convert_f16_to_hf8_sat(val.s5);
    res.data.s6 = _intel_convert_f16_to_hf8_sat(val.s6);
    res.data.s7 = _intel_convert_f16_to_hf8_sat(val.s7);
    return res;
}
fp8e4m3_t16 __attribute__((overloadable)) _convert_fp8e4m3_t16_sat(half16 val) {
    fp8e4m3_t16 res;
    res.data.s0  = _intel_convert_f16_to_hf8_sat(val.s0);
    res.data.s1  = _intel_convert_f16_to_hf8_sat(val.s1);
    res.data.s2  = _intel_convert_f16_to_hf8_sat(val.s2);
    res.data.s3  = _intel_convert_f16_to_hf8_sat(val.s3);
    res.data.s4  = _intel_convert_f16_to_hf8_sat(val.s4);
    res.data.s5  = _intel_convert_f16_to_hf8_sat(val.s5);
    res.data.s6  = _intel_convert_f16_to_hf8_sat(val.s6);
    res.data.s7  = _intel_convert_f16_to_hf8_sat(val.s7);
    res.data.s8  = _intel_convert_f16_to_hf8_sat(val.s8);
    res.data.s9  = _intel_convert_f16_to_hf8_sat(val.s9);
    res.data.sA = _intel_convert_f16_to_hf8_sat(val.sA);
    res.data.sB = _intel_convert_f16_to_hf8_sat(val.sB);
    res.data.sC = _intel_convert_f16_to_hf8_sat(val.sC);
    res.data.sD = _intel_convert_f16_to_hf8_sat(val.sD);
    res.data.sE = _intel_convert_f16_to_hf8_sat(val.sE);
    res.data.sF = _intel_convert_f16_to_hf8_sat(val.sF);
    return res;
}

fp8e8m0_t __attribute__((overloadable)) _convert_fp8e8m0_t(float val) {
    fp8e8m0_t res;
    res.data = _intel_convert_f32_fo_e8m0(val);
    return res;
}
fp8e8m0_t1 __attribute__((overloadable)) _convert_fp8e8m0_t1(float val[1]) {
    fp8e8m0_t1 res;
    res.data = _intel_convert_f32_fo_e8m0(val[0]);
    return res;
}
fp8e8m0_t2 __attribute__((overloadable)) _convert_fp8e8m0_t2(float2 val) {
    fp8e8m0_t2 res;
    res.data.s0 = _intel_convert_f32_fo_e8m0(val.x);
    res.data.s1 = _intel_convert_f32_fo_e8m0(val.y);
    return res;
}
fp8e8m0_t3 __attribute__((overloadable)) _convert_fp8e8m0_t3(float3 val) {
    fp8e8m0_t3 res;
    res.data.s0 = _intel_convert_f32_fo_e8m0(val.x);
    res.data.s1 = _intel_convert_f32_fo_e8m0(val.y);
    res.data.s2 = _intel_convert_f32_fo_e8m0(val.z);
    return res;
}
fp8e8m0_t4 __attribute__((overloadable)) _convert_fp8e8m0_t4(float4 val) {
    fp8e8m0_t4 res;
    res.data.s0 = _intel_convert_f32_fo_e8m0(val.x);
    res.data.s1 = _intel_convert_f32_fo_e8m0(val.y);
    res.data.s2 = _intel_convert_f32_fo_e8m0(val.z);
    res.data.s3 = _intel_convert_f32_fo_e8m0(val.w);
    return res;
}
fp8e8m0_t8 __attribute__((overloadable)) _convert_fp8e8m0_t8(float8 val) {
    fp8e8m0_t8 res;
    res.data.s0 = _intel_convert_f32_fo_e8m0(val.s0);
    res.data.s1 = _intel_convert_f32_fo_e8m0(val.s1);
    res.data.s2 = _intel_convert_f32_fo_e8m0(val.s2);
    res.data.s3 = _intel_convert_f32_fo_e8m0(val.s3);
    res.data.s4 = _intel_convert_f32_fo_e8m0(val.s4);
    res.data.s5 = _intel_convert_f32_fo_e8m0(val.s5);
    res.data.s6 = _intel_convert_f32_fo_e8m0(val.s6);
    res.data.s7 = _intel_convert_f32_fo_e8m0(val.s7);
    return res;
}
fp8e8m0_t16 __attribute__((overloadable)) _convert_fp8e8m0_t16(float16 val) {
    fp8e8m0_t16 res;
    res.data.s0  = _intel_convert_f32_fo_e8m0(val.s0);
    res.data.s1  = _intel_convert_f32_fo_e8m0(val.s1);
    res.data.s2  = _intel_convert_f32_fo_e8m0(val.s2);
    res.data.s3  = _intel_convert_f32_fo_e8m0(val.s3);
    res.data.s4  = _intel_convert_f32_fo_e8m0(val.s4);
    res.data.s5  = _intel_convert_f32_fo_e8m0(val.s5);
    res.data.s6  = _intel_convert_f32_fo_e8m0(val.s6);
    res.data.s7  = _intel_convert_f32_fo_e8m0(val.s7);
    res.data.s8  = _intel_convert_f32_fo_e8m0(val.s8);
    res.data.s9  = _intel_convert_f32_fo_e8m0(val.s9);
    res.data.sA = _intel_convert_f32_fo_e8m0(val.sA);
    res.data.sB = _intel_convert_f32_fo_e8m0(val.sB);
    res.data.sC = _intel_convert_f32_fo_e8m0(val.sC);
    res.data.sD = _intel_convert_f32_fo_e8m0(val.sD);
    res.data.sE = _intel_convert_f32_fo_e8m0(val.sE);
    res.data.sF = _intel_convert_f32_fo_e8m0(val.sF);
    return res;
}

fp8e8m0_t __attribute__((overloadable)) _convert_fp8e8m0_t_sat(float val) {
    fp8e8m0_t res;
    res.data = _intel_convert_f32_fo_e8m0_sat(val);
    return res;
}
fp8e8m0_t1 __attribute__((overloadable)) _convert_fp8e8m0_t1_sat(float val[1]) {
    fp8e8m0_t1 res;
    res.data = _intel_convert_f32_fo_e8m0_sat(val[0]);
    return res;
}
fp8e8m0_t2 __attribute__((overloadable)) _convert_fp8e8m0_t2_sat(float2 val) {
    fp8e8m0_t2 res;
    res.data.s0 = _intel_convert_f32_fo_e8m0_sat(val.x);
    res.data.s1 = _intel_convert_f32_fo_e8m0_sat(val.y);
    return res;
}
fp8e8m0_t3 __attribute__((overloadable)) _convert_fp8e8m0_t3_sat(float3 val) {
    fp8e8m0_t3 res;
    res.data.s0 = _intel_convert_f32_fo_e8m0_sat(val.x);
    res.data.s1 = _intel_convert_f32_fo_e8m0_sat(val.y);
    res.data.s2 = _intel_convert_f32_fo_e8m0_sat(val.z);
    return res;
}
fp8e8m0_t4 __attribute__((overloadable)) _convert_fp8e8m0_t4_sat(float4 val) {
    fp8e8m0_t4 res;
    res.data.s0 = _intel_convert_f32_fo_e8m0_sat(val.x);
    res.data.s1 = _intel_convert_f32_fo_e8m0_sat(val.y);
    res.data.s2 = _intel_convert_f32_fo_e8m0_sat(val.z);
    res.data.s3 = _intel_convert_f32_fo_e8m0_sat(val.w);
    return res;
}
fp8e8m0_t8 __attribute__((overloadable)) _convert_fp8e8m0_t8_sat(float8 val) {
    fp8e8m0_t8 res;
    res.data.s0 = _intel_convert_f32_fo_e8m0_sat(val.s0);
    res.data.s1 = _intel_convert_f32_fo_e8m0_sat(val.s1);
    res.data.s2 = _intel_convert_f32_fo_e8m0_sat(val.s2);
    res.data.s3 = _intel_convert_f32_fo_e8m0_sat(val.s3);
    res.data.s4 = _intel_convert_f32_fo_e8m0_sat(val.s4);
    res.data.s5 = _intel_convert_f32_fo_e8m0_sat(val.s5);
    res.data.s6 = _intel_convert_f32_fo_e8m0_sat(val.s6);
    res.data.s7 = _intel_convert_f32_fo_e8m0_sat(val.s7);
    return res;
}
fp8e8m0_t16 __attribute__((overloadable)) _convert_fp8e8m0_t16_sat(float16 val) {
    fp8e8m0_t16 res;
    res.data.s0  = _intel_convert_f32_fo_e8m0_sat(val.s0);
    res.data.s1  = _intel_convert_f32_fo_e8m0_sat(val.s1);
    res.data.s2  = _intel_convert_f32_fo_e8m0_sat(val.s2);
    res.data.s3  = _intel_convert_f32_fo_e8m0_sat(val.s3);
    res.data.s4  = _intel_convert_f32_fo_e8m0_sat(val.s4);
    res.data.s5  = _intel_convert_f32_fo_e8m0_sat(val.s5);
    res.data.s6  = _intel_convert_f32_fo_e8m0_sat(val.s6);
    res.data.s7  = _intel_convert_f32_fo_e8m0_sat(val.s7);
    res.data.s8  = _intel_convert_f32_fo_e8m0_sat(val.s8);
    res.data.s9  = _intel_convert_f32_fo_e8m0_sat(val.s9);
    res.data.sA = _intel_convert_f32_fo_e8m0_sat(val.sA);
    res.data.sB = _intel_convert_f32_fo_e8m0_sat(val.sB);
    res.data.sC = _intel_convert_f32_fo_e8m0_sat(val.sC);
    res.data.sD = _intel_convert_f32_fo_e8m0_sat(val.sD);
    res.data.sE = _intel_convert_f32_fo_e8m0_sat(val.sE);
    res.data.sF = _intel_convert_f32_fo_e8m0_sat(val.sF);
    return res;
}

fp8e5m2_t __attribute__((overloadable)) as_fp8e5m2_t(uchar val) {
    fp8e5m2_t res;
    res.data = val;
    return res;
}
fp8e5m2_t1 __attribute__((overloadable)) as_fp8e5m2_t1(uchar val[1]) {
    fp8e5m2_t1 res;
    res.data = val[0];
    return res;
}
fp8e5m2_t2 __attribute__((overloadable)) as_fp8e5m2_t2(uchar2 val) {
    fp8e5m2_t2 res;
    res.data.s0 = val.x;
    res.data.s1 = val.y;
    return res;
}
fp8e5m2_t3 __attribute__((overloadable)) as_fp8e5m2_t3(uchar3 val) {
    fp8e5m2_t3 res;
    res.data.s0 = val.x;
    res.data.s1 = val.y;
    res.data.s2 = val.z;
    return res;
}
fp8e5m2_t4 __attribute__((overloadable)) as_fp8e5m2_t4(uchar4 val) {
    fp8e5m2_t4 res;
    res.data.s0 = val.x;
    res.data.s1 = val.y;
    res.data.s2 = val.z;
    res.data.s3 = val.w;
    return res;
}
fp8e5m2_t8 __attribute__((overloadable)) as_fp8e5m2_t8(uchar8 val) {
    fp8e5m2_t8 res;
    res.data.s0 = val.s0;
    res.data.s1 = val.s1;
    res.data.s2 = val.s2;
    res.data.s3 = val.s3;
    res.data.s4 = val.s4;
    res.data.s5 = val.s5;
    res.data.s6 = val.s6;
    res.data.s7 = val.s7;
    return res;
}
fp8e5m2_t16 __attribute__((overloadable)) as_fp8e5m2_t16(uchar16 val) {
    fp8e5m2_t16 res;
    res.data.s0  = val.s0;
    res.data.s1  = val.s1;
    res.data.s2  = val.s2;
    res.data.s3  = val.s3;
    res.data.s4  = val.s4;
    res.data.s5  = val.s5;
    res.data.s6  = val.s6;
    res.data.s7  = val.s7;
    res.data.s8  = val.s8;
    res.data.s9  = val.s9;
    res.data.sA = val.sA;
    res.data.sB = val.sB;
    res.data.sC = val.sC;
    res.data.sD = val.sD;
    res.data.sE = val.sE;
    res.data.sF = val.sF;
    return res;
}

fp8e4m3_t __attribute__((overloadable)) as_fp8e4m3_t(uchar val) {
    fp8e4m3_t res;
    res.data = as_char(val);
    return res;
}
fp8e4m3_t1 __attribute__((overloadable)) as_fp8e4m3_t1(uchar val[1]) {
    fp8e4m3_t1 res;
    res.data = as_char(val[0]);
    return res;
}
fp8e4m3_t2 __attribute__((overloadable)) as_fp8e4m3_t2(uchar2 val) {
    fp8e4m3_t2 res;
    res.data.s0 = as_char(val.x);
    res.data.s1 = as_char(val.y);
    return res;
}
fp8e4m3_t3 __attribute__((overloadable)) as_fp8e4m3_t3(uchar3 val) {
    fp8e4m3_t3 res;
    res.data.s0 = as_char(val.x);
    res.data.s1 = as_char(val.y);
    res.data.s2 = as_char(val.z);
    return res;
}
fp8e4m3_t4 __attribute__((overloadable)) as_fp8e4m3_t4(uchar4 val) {
    fp8e4m3_t4 res;
    res.data.s0 = as_char(val.x);
    res.data.s1 = as_char(val.y);
    res.data.s2 = as_char(val.z);
    res.data.s3 = as_char(val.w);
    return res;
}
fp8e4m3_t8 __attribute__((overloadable)) as_fp8e4m3_t8(uchar8 val) {
    fp8e4m3_t8 res;
    res.data.s0 = as_char(val.s0);
    res.data.s1 = as_char(val.s1);
    res.data.s2 = as_char(val.s2);
    res.data.s3 = as_char(val.s3);
    res.data.s4 = as_char(val.s4);
    res.data.s5 = as_char(val.s5);
    res.data.s6 = as_char(val.s6);
    res.data.s7 = as_char(val.s7);
    return res;
}
fp8e4m3_t16 __attribute__((overloadable)) as_fp8e4m3_t16(uchar16 val) {
    fp8e4m3_t16 res;
    res.data.s0  = as_char(val.s0);
    res.data.s1  = as_char(val.s1);
    res.data.s2  = as_char(val.s2);
    res.data.s3  = as_char(val.s3);
    res.data.s4  = as_char(val.s4);
    res.data.s5  = as_char(val.s5);
    res.data.s6  = as_char(val.s6);
    res.data.s7  = as_char(val.s7);
    res.data.s8  = as_char(val.s8);
    res.data.s9  = as_char(val.s9);
    res.data.sA = as_char(val.sA);
    res.data.sB = as_char(val.sB);
    res.data.sC = as_char(val.sC);
    res.data.sD = as_char(val.sD);
    res.data.sE = as_char(val.sE);
    res.data.sF = as_char(val.sF);
    return res;
}

fp8e8m0_t __attribute__((overloadable)) as_fp8e8m0_t(uchar val) {
    fp8e8m0_t res;
    res.data = val;
    return res;
}
fp8e8m0_t1 __attribute__((overloadable)) as_fp8e8m0_t1(uchar val[1]) {
    fp8e8m0_t1 res;
    res.data = val[0];
    return res;
}
fp8e8m0_t2 __attribute__((overloadable)) as_fp8e8m0_t2(uchar2 val) {
    fp8e8m0_t2 res;
    res.data.s0 = val.x;
    res.data.s1 = val.y;
    return res;
}
fp8e8m0_t3 __attribute__((overloadable)) as_fp8e8m0_t3(uchar3 val) {
    fp8e8m0_t3 res;
    res.data.s0 = val.x;
    res.data.s1 = val.y;
    res.data.s2 = val.z;
    return res;
}
fp8e8m0_t4 __attribute__((overloadable)) as_fp8e8m0_t4(uchar4 val) {
    fp8e8m0_t4 res;
    res.data.s0 = val.x;
    res.data.s1 = val.y;
    res.data.s2 = val.z;
    res.data.s3 = val.w;
    return res;
}
fp8e8m0_t8 __attribute__((overloadable)) as_fp8e8m0_t8(uchar8 val) {
    fp8e8m0_t8 res;
    res.data.s0 = val.s0;
    res.data.s1 = val.s1;
    res.data.s2 = val.s2;
    res.data.s3 = val.s3;
    res.data.s4 = val.s4;
    res.data.s5 = val.s5;
    res.data.s6 = val.s6;
    res.data.s7 = val.s7;
    return res;
}
fp8e8m0_t16 __attribute__((overloadable)) as_fp8e8m0_t16(uchar16 val) {
    fp8e8m0_t16 res;
    res.data.s0  = val.s0;
    res.data.s1  = val.s1;
    res.data.s2  = val.s2;
    res.data.s3  = val.s3;
    res.data.s4  = val.s4;
    res.data.s5  = val.s5;
    res.data.s6  = val.s6;
    res.data.s7  = val.s7;
    res.data.s8  = val.s8;
    res.data.s9  = val.s9;
    res.data.sA = val.sA;
    res.data.sB = val.sB;
    res.data.sC = val.sC;
    res.data.sD = val.sD;
    res.data.sE = val.sE;
    res.data.sF = val.sF;
    return res;
}

uchar __attribute__((overloadable)) _as_uchar(fp8e5m2_t val) {
    return val.data;
}
uchar __attribute__((overloadable)) _as_uchar(fp8e5m2_t1 val) {
    return val.data;
}
uchar2 __attribute__((overloadable)) _as_uchar2(fp8e5m2_t2 val) {
    return (uchar2)(val.data.s0, val.data.s1);
}
uchar3 __attribute__((overloadable)) _as_uchar3(fp8e5m2_t3 val) {
    return (uchar3)(val.data.s0, val.data.s1, val.data.s2);
}
uchar4 __attribute__((overloadable)) _as_uchar4(fp8e5m2_t4 val) {
    return (uchar4)(val.data.s0, val.data.s1, val.data.s2, val.data.s3);
}
uchar8 __attribute__((overloadable)) _as_uchar8(fp8e5m2_t8 val) {
    return (uchar8)(val.data.s0, val.data.s1, val.data.s2, val.data.s3,
                    val.data.s4, val.data.s5, val.data.s6, val.data.s7);
}
uchar16 __attribute__((overloadable)) _as_uchar16(fp8e5m2_t16 val) {
    return (uchar16)(val.data.s0,  val.data.s1,  val.data.s2,  val.data.s3,
                     val.data.s4,  val.data.s5,  val.data.s6,  val.data.s7,
                     val.data.s8,  val.data.s9,  val.data.sA, val.data.sB,
                     val.data.sC, val.data.sD, val.data.sE, val.data.sF);
}

uchar __attribute__((overloadable)) _as_uchar(fp8e4m3_t val) {
    return as_uchar((char)val.data);
}
uchar __attribute__((overloadable)) _as_uchar(fp8e4m3_t1 val) {
    return as_uchar((char)val.data);
}
uchar2 __attribute__((overloadable)) _as_uchar2(fp8e4m3_t2 val) {
    return (uchar2)(as_uchar((char)val.data.s0), as_uchar((char)val.data.s1));
}
uchar3 __attribute__((overloadable)) _as_uchar3(fp8e4m3_t3 val) {
    return (uchar3)(as_uchar((char)val.data.s0), as_uchar((char)val.data.s1), as_uchar((char)val.data.s2));
}
uchar4 __attribute__((overloadable)) _as_uchar4(fp8e4m3_t4 val) {
    return (uchar4)(as_uchar((char)val.data.s0), as_uchar((char)val.data.s1),
                    as_uchar((char)val.data.s2), as_uchar((char)val.data.s3));
}
uchar8 __attribute__((overloadable)) _as_uchar8(fp8e4m3_t8 val) {
    return (uchar8)(as_uchar((char)val.data.s0), as_uchar((char)val.data.s1),
                    as_uchar((char)val.data.s2), as_uchar((char)val.data.s3),
                    as_uchar((char)val.data.s4), as_uchar((char)val.data.s5),
                    as_uchar((char)val.data.s6), as_uchar((char)val.data.s7));
}
uchar16 __attribute__((overloadable)) _as_uchar16(fp8e4m3_t16 val) {
    return (uchar16)(as_uchar((char)val.data.s0),  as_uchar((char)val.data.s1),
                     as_uchar((char)val.data.s2),  as_uchar((char)val.data.s3),
                     as_uchar((char)val.data.s4),  as_uchar((char)val.data.s5),
                     as_uchar((char)val.data.s6),  as_uchar((char)val.data.s7),
                     as_uchar((char)val.data.s8),  as_uchar((char)val.data.s9),
                     as_uchar((char)val.data.sA), as_uchar((char)val.data.sB),
                     as_uchar((char)val.data.sC), as_uchar((char)val.data.sD),
                     as_uchar((char)val.data.sE), as_uchar((char)val.data.sF));
}

uchar __attribute__((overloadable)) _as_uchar(fp8e8m0_t val) {
    return val.data;
}
uchar __attribute__((overloadable)) _as_uchar(fp8e8m0_t1 val) {
    return val.data;
}
uchar2 __attribute__((overloadable)) _as_uchar2(fp8e8m0_t2 val) {
    return (uchar2)(val.data.s0, val.data.s1);
}
uchar3 __attribute__((overloadable)) _as_uchar3(fp8e8m0_t3 val) {
    return (uchar3)(val.data.s0, val.data.s1, val.data.s2);
}
uchar4 __attribute__((overloadable)) _as_uchar4(fp8e8m0_t4 val) {
    return (uchar4)(val.data.s0, val.data.s1, val.data.s2, val.data.s3);
}
uchar8 __attribute__((overloadable)) _as_uchar8(fp8e8m0_t8 val) {
    return (uchar8)(val.data.s0, val.data.s1, val.data.s2, val.data.s3,
                    val.data.s4, val.data.s5, val.data.s6, val.data.s7);
}
uchar16 __attribute__((overloadable)) _as_uchar16(fp8e8m0_t16 val) {
    return (uchar16)(val.data.s0,  val.data.s1,  val.data.s2,  val.data.s3,
                     val.data.s4,  val.data.s5,  val.data.s6,  val.data.s7,
                     val.data.s8,  val.data.s9,  val.data.sA, val.data.sB,
                     val.data.sC, val.data.sD, val.data.sE, val.data.sF);
}
