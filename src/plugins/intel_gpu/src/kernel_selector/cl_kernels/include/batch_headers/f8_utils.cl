// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// TODO: Replace `_intel_convert*` with bultins when ready, current implementations are copied from XeTLA:

#ifndef OV_GPU_OCL_F8_UTILS_H
#define OV_GPU_OCL_F8_UTILS_H

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

uchar _f16_to_hf8_universal(half val, bool is_saturation) {
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

uchar _intel_convert_f16_to_hf8(half val) {
    return _f16_to_hf8_universal(val, false);
}

uchar _intel_convert_f16_to_hf8_sat(half val) {
    return _f16_to_hf8_universal(val, true);
}

half _intel_convert_hf8_to_f16(uchar val) {
    const short exp_bias = 7;
    uchar data = val;
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

uchar _intel_convert_f32_to_e8m0(float val) {
    uint val_uint = as_uint(val);
    return (uchar)((val_uint >> 23) & 0xFF);
}

uchar _intel_convert_f32_to_e8m0_sat(float val) {
    return _intel_convert_f32_to_e8m0(val);
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

inline uchar2 _convert_fp8e4m3_t2_as_uchar2(half2 source) {
    return (uchar2)(_intel_convert_f16_to_hf8(source.s0),
                     _intel_convert_f16_to_hf8(source.s1));
}

inline uchar3 _convert_fp8e4m3_t3_as_uchar3(half3 source) {
    return (uchar3)(_intel_convert_f16_to_hf8(source.s0),
                     _intel_convert_f16_to_hf8(source.s1),
                     _intel_convert_f16_to_hf8(source.s2));
}

inline uchar4 _convert_fp8e4m3_t4_as_uchar4(half4 source) {
    return (uchar4)(_intel_convert_f16_to_hf8(source.s0),
                     _intel_convert_f16_to_hf8(source.s1),
                     _intel_convert_f16_to_hf8(source.s2),
                     _intel_convert_f16_to_hf8(source.s3));
}

inline uchar8 _convert_fp8e4m3_t8_as_uchar8(half8 source) {
    return (uchar8)(_intel_convert_f16_to_hf8(source.s0),
                     _intel_convert_f16_to_hf8(source.s1),
                     _intel_convert_f16_to_hf8(source.s2),
                     _intel_convert_f16_to_hf8(source.s3),
                     _intel_convert_f16_to_hf8(source.s4),
                     _intel_convert_f16_to_hf8(source.s5),
                     _intel_convert_f16_to_hf8(source.s6),
                     _intel_convert_f16_to_hf8(source.s7));
}

inline uchar16 _convert_fp8e4m3_t16_as_uchar16(half16 source) {
    return (uchar16)(_intel_convert_f16_to_hf8(source.s0),
                      _intel_convert_f16_to_hf8(source.s1),
                      _intel_convert_f16_to_hf8(source.s2),
                      _intel_convert_f16_to_hf8(source.s3),
                      _intel_convert_f16_to_hf8(source.s4),
                      _intel_convert_f16_to_hf8(source.s5),
                      _intel_convert_f16_to_hf8(source.s6),
                      _intel_convert_f16_to_hf8(source.s7),
                      _intel_convert_f16_to_hf8(source.s8),
                      _intel_convert_f16_to_hf8(source.s9),
                      _intel_convert_f16_to_hf8(source.sa),
                      _intel_convert_f16_to_hf8(source.sb),
                      _intel_convert_f16_to_hf8(source.sc),
                      _intel_convert_f16_to_hf8(source.sd),
                      _intel_convert_f16_to_hf8(source.se),
                      _intel_convert_f16_to_hf8(source.sf));
}

inline uchar2 _convert_fp8e4m3_t2_as_uchar2_sat(half2 source) {
    return (uchar2)(_intel_convert_f16_to_hf8_sat(source.s0),
                     _intel_convert_f16_to_hf8_sat(source.s1));
}

inline uchar3 _convert_fp8e4m3_t3_as_uchar3_sat(half3 source) {
    return (uchar3)(_intel_convert_f16_to_hf8_sat(source.s0),
                     _intel_convert_f16_to_hf8_sat(source.s1),
                     _intel_convert_f16_to_hf8_sat(source.s2));
}

inline uchar4 _convert_fp8e4m3_t4_as_uchar4_sat(half4 source) {
    return (uchar4)(_intel_convert_f16_to_hf8_sat(source.s0),
                     _intel_convert_f16_to_hf8_sat(source.s1),
                     _intel_convert_f16_to_hf8_sat(source.s2),
                     _intel_convert_f16_to_hf8_sat(source.s3));
}

inline uchar8 _convert_fp8e4m3_t8_as_uchar8_sat(half8 source) {
    return (uchar8)(_intel_convert_f16_to_hf8_sat(source.s0),
                     _intel_convert_f16_to_hf8_sat(source.s1),
                     _intel_convert_f16_to_hf8_sat(source.s2),
                     _intel_convert_f16_to_hf8_sat(source.s3),
                     _intel_convert_f16_to_hf8_sat(source.s4),
                     _intel_convert_f16_to_hf8_sat(source.s5),
                     _intel_convert_f16_to_hf8_sat(source.s6),
                     _intel_convert_f16_to_hf8_sat(source.s7));
}

inline uchar16 _convert_fp8e4m3_t16_as_uchar16_sat(half16 source) {
    return (uchar16)(_intel_convert_f16_to_hf8_sat(source.s0),
                      _intel_convert_f16_to_hf8_sat(source.s1),
                      _intel_convert_f16_to_hf8_sat(source.s2),
                      _intel_convert_f16_to_hf8_sat(source.s3),
                      _intel_convert_f16_to_hf8_sat(source.s4),
                      _intel_convert_f16_to_hf8_sat(source.s5),
                      _intel_convert_f16_to_hf8_sat(source.s6),
                      _intel_convert_f16_to_hf8_sat(source.s7),
                      _intel_convert_f16_to_hf8_sat(source.s8),
                      _intel_convert_f16_to_hf8_sat(source.s9),
                      _intel_convert_f16_to_hf8_sat(source.sa),
                      _intel_convert_f16_to_hf8_sat(source.sb),
                      _intel_convert_f16_to_hf8_sat(source.sc),
                      _intel_convert_f16_to_hf8_sat(source.sd),
                      _intel_convert_f16_to_hf8_sat(source.se),
                      _intel_convert_f16_to_hf8_sat(source.sf));
}

inline half2 _convert_as_fp8e4m3_t2_half2(uchar2 source) {
    return (half2)(_intel_convert_hf8_to_f16(source.s0),
                    _intel_convert_hf8_to_f16(source.s1));
}

inline half3 _convert_as_fp8e4m3_t3_half3(uchar3 source) {
    return (half3)(_intel_convert_hf8_to_f16(source.s0),
                    _intel_convert_hf8_to_f16(source.s1),
                    _intel_convert_hf8_to_f16(source.s2));
}

inline half4 _convert_as_fp8e4m3_t4_half4(uchar4 source) {
    return (half4)(_intel_convert_hf8_to_f16(source.s0),
                    _intel_convert_hf8_to_f16(source.s1),
                    _intel_convert_hf8_to_f16(source.s2),
                    _intel_convert_hf8_to_f16(source.s3));
}

inline half8 _convert_as_fp8e4m3_t8_half8(uchar8 source) {
    return (half8)(_intel_convert_hf8_to_f16(source.s0),
                    _intel_convert_hf8_to_f16(source.s1),
                    _intel_convert_hf8_to_f16(source.s2),
                    _intel_convert_hf8_to_f16(source.s3),
                    _intel_convert_hf8_to_f16(source.s4),
                    _intel_convert_hf8_to_f16(source.s5),
                    _intel_convert_hf8_to_f16(source.s6),
                    _intel_convert_hf8_to_f16(source.s7));
}

inline half16 _convert_as_fp8e4m3_t16_half16(uchar16 source) {
    return (half16)(_intel_convert_hf8_to_f16(source.s0),
                     _intel_convert_hf8_to_f16(source.s1),
                     _intel_convert_hf8_to_f16(source.s2),
                     _intel_convert_hf8_to_f16(source.s3),
                     _intel_convert_hf8_to_f16(source.s4),
                     _intel_convert_hf8_to_f16(source.s5),
                     _intel_convert_hf8_to_f16(source.s6),
                     _intel_convert_hf8_to_f16(source.s7),
                     _intel_convert_hf8_to_f16(source.s8),
                     _intel_convert_hf8_to_f16(source.s9),
                     _intel_convert_hf8_to_f16(source.sa),
                     _intel_convert_hf8_to_f16(source.sb),
                     _intel_convert_hf8_to_f16(source.sc),
                     _intel_convert_hf8_to_f16(source.sd),
                     _intel_convert_hf8_to_f16(source.se),
                     _intel_convert_hf8_to_f16(source.sf));
}

inline uchar2 _convert_fp8e5m2_t2_as_uchar2(half2 source) {
    return (uchar2)(_intel_convert_f16_to_bf8(source.s0),
                     _intel_convert_f16_to_bf8(source.s1));
}

inline uchar3 _convert_fp8e5m2_t3_as_uchar3(half3 source) {
    return (uchar3)(_intel_convert_f16_to_bf8(source.s0),
                     _intel_convert_f16_to_bf8(source.s1),
                     _intel_convert_f16_to_bf8(source.s2));
}

inline uchar4 _convert_fp8e5m2_t4_as_uchar4(half4 source) {
    return (uchar4)(_intel_convert_f16_to_bf8(source.s0),
                     _intel_convert_f16_to_bf8(source.s1),
                     _intel_convert_f16_to_bf8(source.s2),
                     _intel_convert_f16_to_bf8(source.s3));
}

inline uchar8 _convert_fp8e5m2_t8_as_uchar8(half8 source) {
    return (uchar8)(_intel_convert_f16_to_bf8(source.s0),
                     _intel_convert_f16_to_bf8(source.s1),
                     _intel_convert_f16_to_bf8(source.s2),
                     _intel_convert_f16_to_bf8(source.s3),
                     _intel_convert_f16_to_bf8(source.s4),
                     _intel_convert_f16_to_bf8(source.s5),
                     _intel_convert_f16_to_bf8(source.s6),
                     _intel_convert_f16_to_bf8(source.s7));
}

inline uchar16 _convert_fp8e5m2_t16_as_uchar16(half16 source) {
    return (uchar16)(_intel_convert_f16_to_bf8(source.s0),
                      _intel_convert_f16_to_bf8(source.s1),
                      _intel_convert_f16_to_bf8(source.s2),
                      _intel_convert_f16_to_bf8(source.s3),
                      _intel_convert_f16_to_bf8(source.s4),
                      _intel_convert_f16_to_bf8(source.s5),
                      _intel_convert_f16_to_bf8(source.s6),
                      _intel_convert_f16_to_bf8(source.s7),
                      _intel_convert_f16_to_bf8(source.s8),
                      _intel_convert_f16_to_bf8(source.s9),
                      _intel_convert_f16_to_bf8(source.sa),
                      _intel_convert_f16_to_bf8(source.sb),
                      _intel_convert_f16_to_bf8(source.sc),
                      _intel_convert_f16_to_bf8(source.sd),
                      _intel_convert_f16_to_bf8(source.se),
                      _intel_convert_f16_to_bf8(source.sf));
}

inline uchar2 _convert_fp8e5m2_t2_as_uchar2_sat(half2 source) {
    return (uchar2)(_intel_convert_f16_to_bf8_sat(source.s0),
                     _intel_convert_f16_to_bf8_sat(source.s1));
}

inline uchar3 _convert_fp8e5m2_t3_as_uchar3_sat(half3 source) {
    return (uchar3)(_intel_convert_f16_to_bf8_sat(source.s0),
                     _intel_convert_f16_to_bf8_sat(source.s1),
                     _intel_convert_f16_to_bf8_sat(source.s2));
}

inline uchar4 _convert_fp8e5m2_t4_as_uchar4_sat(half4 source) {
    return (uchar4)(_intel_convert_f16_to_bf8_sat(source.s0),
                     _intel_convert_f16_to_bf8_sat(source.s1),
                     _intel_convert_f16_to_bf8_sat(source.s2),
                     _intel_convert_f16_to_bf8_sat(source.s3));
}

inline uchar8 _convert_fp8e5m2_t8_as_uchar8_sat(half8 source) {
    return (uchar8)(_intel_convert_f16_to_bf8_sat(source.s0),
                     _intel_convert_f16_to_bf8_sat(source.s1),
                     _intel_convert_f16_to_bf8_sat(source.s2),
                     _intel_convert_f16_to_bf8_sat(source.s3),
                     _intel_convert_f16_to_bf8_sat(source.s4),
                     _intel_convert_f16_to_bf8_sat(source.s5),
                     _intel_convert_f16_to_bf8_sat(source.s6),
                     _intel_convert_f16_to_bf8_sat(source.s7));
}

inline uchar16 _convert_fp8e5m2_t16_as_uchar16_sat(half16 source) {
    return (uchar16)(_intel_convert_f16_to_bf8_sat(source.s0),
                      _intel_convert_f16_to_bf8_sat(source.s1),
                      _intel_convert_f16_to_bf8_sat(source.s2),
                      _intel_convert_f16_to_bf8_sat(source.s3),
                      _intel_convert_f16_to_bf8_sat(source.s4),
                      _intel_convert_f16_to_bf8_sat(source.s5),
                      _intel_convert_f16_to_bf8_sat(source.s6),
                      _intel_convert_f16_to_bf8_sat(source.s7),
                      _intel_convert_f16_to_bf8_sat(source.s8),
                      _intel_convert_f16_to_bf8_sat(source.s9),
                      _intel_convert_f16_to_bf8_sat(source.sa),
                      _intel_convert_f16_to_bf8_sat(source.sb),
                      _intel_convert_f16_to_bf8_sat(source.sc),
                      _intel_convert_f16_to_bf8_sat(source.sd),
                      _intel_convert_f16_to_bf8_sat(source.se),
                      _intel_convert_f16_to_bf8_sat(source.sf));
}

inline half2 _convert_as_fp8e5m2_t2_half2(uchar2 source) {
    return (half2)(_intel_convert_bf8_to_f16(source.s0),
                    _intel_convert_bf8_to_f16(source.s1));
}

inline half3 _convert_as_fp8e5m2_t3_half3(uchar3 source) {
    return (half3)(_intel_convert_bf8_to_f16(source.s0),
                    _intel_convert_bf8_to_f16(source.s1),
                    _intel_convert_bf8_to_f16(source.s2));
}

inline half4 _convert_as_fp8e5m2_t4_half4(uchar4 source) {
    return (half4)(_intel_convert_bf8_to_f16(source.s0),
                    _intel_convert_bf8_to_f16(source.s1),
                    _intel_convert_bf8_to_f16(source.s2),
                    _intel_convert_bf8_to_f16(source.s3));
}

inline half8 _convert_as_fp8e5m2_t8_half8(uchar8 source) {
    return (half8)(_intel_convert_bf8_to_f16(source.s0),
                    _intel_convert_bf8_to_f16(source.s1),
                    _intel_convert_bf8_to_f16(source.s2),
                    _intel_convert_bf8_to_f16(source.s3),
                    _intel_convert_bf8_to_f16(source.s4),
                    _intel_convert_bf8_to_f16(source.s5),
                    _intel_convert_bf8_to_f16(source.s6),
                    _intel_convert_bf8_to_f16(source.s7));
}

inline half16 _convert_as_fp8e5m2_t16_half16(uchar16 source) {
    return (half16)(_intel_convert_bf8_to_f16(source.s0),
                     _intel_convert_bf8_to_f16(source.s1),
                     _intel_convert_bf8_to_f16(source.s2),
                     _intel_convert_bf8_to_f16(source.s3),
                     _intel_convert_bf8_to_f16(source.s4),
                     _intel_convert_bf8_to_f16(source.s5),
                     _intel_convert_bf8_to_f16(source.s6),
                     _intel_convert_bf8_to_f16(source.s7),
                     _intel_convert_bf8_to_f16(source.s8),
                     _intel_convert_bf8_to_f16(source.s9),
                     _intel_convert_bf8_to_f16(source.sa),
                     _intel_convert_bf8_to_f16(source.sb),
                     _intel_convert_bf8_to_f16(source.sc),
                     _intel_convert_bf8_to_f16(source.sd),
                     _intel_convert_bf8_to_f16(source.se),
                     _intel_convert_bf8_to_f16(source.sf));
}

inline uchar2 _convert_fp8e8m0_t2_as_uchar2(float2 source) {
    return (uchar2)(_intel_convert_f32_to_e8m0(source.s0),
                     _intel_convert_f32_to_e8m0(source.s1));
}

inline uchar3 _convert_fp8e8m0_t3_as_uchar3(float3 source) {
    return (uchar3)(_intel_convert_f32_to_e8m0(source.s0),
                     _intel_convert_f32_to_e8m0(source.s1),
                     _intel_convert_f32_to_e8m0(source.s2));
}

inline uchar4 _convert_fp8e8m0_t4_as_uchar4(float4 source) {
    return (uchar4)(_intel_convert_f32_to_e8m0(source.s0),
                     _intel_convert_f32_to_e8m0(source.s1),
                     _intel_convert_f32_to_e8m0(source.s2),
                     _intel_convert_f32_to_e8m0(source.s3));
}

inline uchar8 _convert_fp8e8m0_t8_as_uchar8(float8 source) {
    return (uchar8)(_intel_convert_f32_to_e8m0(source.s0),
                     _intel_convert_f32_to_e8m0(source.s1),
                     _intel_convert_f32_to_e8m0(source.s2),
                     _intel_convert_f32_to_e8m0(source.s3),
                     _intel_convert_f32_to_e8m0(source.s4),
                     _intel_convert_f32_to_e8m0(source.s5),
                     _intel_convert_f32_to_e8m0(source.s6),
                     _intel_convert_f32_to_e8m0(source.s7));
}

inline uchar16 _convert_fp8e8m0_t16_as_uchar16(float16 source) {
    return (uchar16)(_intel_convert_f32_to_e8m0(source.s0),
                      _intel_convert_f32_to_e8m0(source.s1),
                      _intel_convert_f32_to_e8m0(source.s2),
                      _intel_convert_f32_to_e8m0(source.s3),
                      _intel_convert_f32_to_e8m0(source.s4),
                      _intel_convert_f32_to_e8m0(source.s5),
                      _intel_convert_f32_to_e8m0(source.s6),
                      _intel_convert_f32_to_e8m0(source.s7),
                      _intel_convert_f32_to_e8m0(source.s8),
                      _intel_convert_f32_to_e8m0(source.s9),
                      _intel_convert_f32_to_e8m0(source.sa),
                      _intel_convert_f32_to_e8m0(source.sb),
                      _intel_convert_f32_to_e8m0(source.sc),
                      _intel_convert_f32_to_e8m0(source.sd),
                      _intel_convert_f32_to_e8m0(source.se),
                      _intel_convert_f32_to_e8m0(source.sf));
}

inline float2 _convert_as_fp8e8m0_t2_float2(uchar2 source) {
    return (float2)(_intel_convert_e8m0_to_f32(source.s0),
                    _intel_convert_e8m0_to_f32(source.s1));
}

inline float3 _convert_as_fp8e8m0_t3_float3(uchar3 source) {
    return (float3)(_intel_convert_e8m0_to_f32(source.s0),
                    _intel_convert_e8m0_to_f32(source.s1),
                    _intel_convert_e8m0_to_f32(source.s2));
}

inline float4 _convert_as_fp8e8m0_t4_float4(uchar4 source) {
    return (float4)(_intel_convert_e8m0_to_f32(source.s0),
                    _intel_convert_e8m0_to_f32(source.s1),
                    _intel_convert_e8m0_to_f32(source.s2),
                    _intel_convert_e8m0_to_f32(source.s3));
}

inline float8 _convert_as_fp8e8m0_t8_float8(uchar8 source) {
    return (float8)(_intel_convert_e8m0_to_f32(source.s0),
                    _intel_convert_e8m0_to_f32(source.s1),
                    _intel_convert_e8m0_to_f32(source.s2),
                    _intel_convert_e8m0_to_f32(source.s3),
                    _intel_convert_e8m0_to_f32(source.s4),
                    _intel_convert_e8m0_to_f32(source.s5),
                    _intel_convert_e8m0_to_f32(source.s6),
                    _intel_convert_e8m0_to_f32(source.s7));
}

inline float16 _convert_as_fp8e8m0_t16_float16(uchar16 source) {
    return (float16)(_intel_convert_e8m0_to_f32(source.s0),
                     _intel_convert_e8m0_to_f32(source.s1),
                     _intel_convert_e8m0_to_f32(source.s2),
                     _intel_convert_e8m0_to_f32(source.s3),
                     _intel_convert_e8m0_to_f32(source.s4),
                     _intel_convert_e8m0_to_f32(source.s5),
                     _intel_convert_e8m0_to_f32(source.s6),
                     _intel_convert_e8m0_to_f32(source.s7),
                     _intel_convert_e8m0_to_f32(source.s8),
                     _intel_convert_e8m0_to_f32(source.s9),
                     _intel_convert_e8m0_to_f32(source.sa),
                     _intel_convert_e8m0_to_f32(source.sb),
                     _intel_convert_e8m0_to_f32(source.sc),
                     _intel_convert_e8m0_to_f32(source.sd),
                     _intel_convert_e8m0_to_f32(source.se),
                     _intel_convert_e8m0_to_f32(source.sf));
}

#define CONVERT_F8E4M3_AS_UCHAR(val, size) CAT(_convert_fp8e4m3_t, CAT(size, CAT(_as_uchar, size)))(val)
#define CONVERT_F8E4M3_AS_UCHAR_SAT(val, size) CAT(_convert_fp8e4m3_t, CAT(size, CAT(_as_uchar, CAT(size, _sat))))(val)
#define CONVERT_AS_F8E4M3_HALF(val, size)  CAT(_convert_as_fp8e4m3_t, CAT(size, CAT(_half, size)))(val)
#define CONVERT_F8E5M2_AS_UCHAR(val, size) CAT(_convert_fp8e5m2_t, CAT(size, CAT(_as_uchar, size)))(val)
#define CONVERT_F8E5M2_AS_UCHAR_SAT(val, size) CAT(_convert_fp8e5m2_t, CAT(size, CAT(_as_uchar, CAT(size, _sat))))(val)
#define CONVERT_AS_F8E5M2_HALF(val, size)  CAT(_convert_as_fp8e5m2_t, CAT(size, CAT(_half, size)))(val)
#define CONVERT_F8E8M0_AS_UCHAR(val, size) CAT(_convert_fp8e8m0_t, CAT(size, CAT(_as_uchar, size)))(val)
#define CONVERT_AS_F8E8M0_FLOAT(val, size)  CAT(_convert_as_fp8e8m0_t, CAT(size, CAT(_float, size)))(val)

#endif
