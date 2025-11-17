// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// TODO: Replace `_intel_convert*` with bultins when ready, current implementations are copied from XeTLA:

uchar _f16_to_fp4_universal(half val) {
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
    const uchar  nan = 0x0;
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
        //std::cout << "ZERO: ";
        dst_val = 0;
        // if (src_exp_unbiased == -2 && src_mant > 0) // if larger then 0.25 round to 0.5
        if (src_exp == 0xD && src_mant & 0x3FF) // if larger then 0.25 round to 0.5
            dst_val = 0x1;
    } else if (is_denorm) {
        //std::cout << "DENORM: ";
        dst_val = 0x1;
        // if (src_exp_unbiased == -1 && src_mant >= 512) // if larger then 0.75 round to 1.0
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
    return (src_sign << (exp_size + mant_size) | dst_val) & 0xF;
}

uchar _intel_convert_f16_to_f4(half val) {
    return _f16_to_fp4_universal(val);
}

uchar _intel_convert_f16_to_f4_sat(half val) {
    return _f16_to_fp4_universal(val);
}

half _intel_convert_fp4_to_f16(uchar val){
        ushort LUT[16] = {0x0000, 0x3800, 0x3c00, 0x3e00, 0x4000,
                0x4200, 0x4400, 0x4600, 0x8000, 0xb800, 0xbc00, 0xbe00, 0xc000,
                0xc200, 0xc400, 0xc600};
        ushort idx = val & 0xf;
        ushort looked_val = LUT[idx];
        half looked_val_fp16 = as_half(looked_val);
        return looked_val_fp16;
}

typedef struct fp4e2m1_t { uchar data; } fp4e2m1_t;  // f4
typedef struct fp4e2m1_t1 { uchar data[1]; } fp4e2m1_t1;
typedef struct fp4e2m1_t2 { uchar data[2]; } fp4e2m1_t2;
typedef struct fp4e2m1_t3 { uchar data[3]; } fp4e2m1_t3;
typedef struct fp4e2m1_t4 { uchar data[4]; } fp4e2m1_t4;
typedef struct fp4e2m1_t8 { uchar data[8]; } fp4e2m1_t8;
typedef struct fp4e2m1_t16 { uchar data[16]; } fp4e2m1_t16;

half __attribute__((overloadable)) _convert_half(fp4e2m1_t val) {
    return _intel_convert_fp4_to_f16(val.data);
}
half __attribute__((overloadable)) _convert_half(fp4e2m1_t1 val) {
    return _intel_convert_fp4_to_f16(val.data[0]);
}
half2 __attribute__((overloadable)) _convert_half2(fp4e2m1_t2 val) {
    return (half2)(_intel_convert_fp4_to_f16(val.data[0]), _intel_convert_fp4_to_f16(val.data[1]));
}
half3 __attribute__((overloadable)) _convert_half3(fp4e2m1_t3 val) {
    return (half3)(_intel_convert_fp4_to_f16(val.data[0]), _intel_convert_fp4_to_f16(val.data[1]), _intel_convert_fp4_to_f16(val.data[2]));
}
half4 __attribute__((overloadable)) _convert_half4(fp4e2m1_t4 val) {
    return (half4)(_intel_convert_fp4_to_f16(val.data[0]), _intel_convert_fp4_to_f16(val.data[1]), 
                  _intel_convert_fp4_to_f16(val.data[2]), _intel_convert_fp4_to_f16(val.data[3]));
}
half8 __attribute__((overloadable)) _convert_half8(fp4e2m1_t8 val) {
    return (half8)(_intel_convert_fp4_to_f16(val.data[0]), _intel_convert_fp4_to_f16(val.data[1]),
                   _intel_convert_fp4_to_f16(val.data[2]), _intel_convert_fp4_to_f16(val.data[3]),
                   _intel_convert_fp4_to_f16(val.data[4]), _intel_convert_fp4_to_f16(val.data[5]),
                   _intel_convert_fp4_to_f16(val.data[6]), _intel_convert_fp4_to_f16(val.data[7]));
}
half16 __attribute__((overloadable)) _convert_half16(fp4e2m1_t16 val) {
    return (half16)(
        _intel_convert_fp4_to_f16(val.data[0]),  _intel_convert_fp4_to_f16(val.data[1]),
        _intel_convert_fp4_to_f16(val.data[2]),  _intel_convert_fp4_to_f16(val.data[3]),
        _intel_convert_fp4_to_f16(val.data[4]),  _intel_convert_fp4_to_f16(val.data[5]),
        _intel_convert_fp4_to_f16(val.data[6]),  _intel_convert_fp4_to_f16(val.data[7]),
        _intel_convert_fp4_to_f16(val.data[8]),  _intel_convert_fp4_to_f16(val.data[9]),
        _intel_convert_fp4_to_f16(val.data[10]), _intel_convert_fp4_to_f16(val.data[11]),
        _intel_convert_fp4_to_f16(val.data[12]), _intel_convert_fp4_to_f16(val.data[13]),
        _intel_convert_fp4_to_f16(val.data[14]), _intel_convert_fp4_to_f16(val.data[15])
    );
}

float __attribute__((overloadable)) _convert_float(fp4e2m1_t val) {
    return (float)_intel_convert_fp4_to_f16(val.data);
}
float __attribute__((overloadable)) _convert_float(fp4e2m1_t1 val) {
    return (float)_intel_convert_fp4_to_f16(val.data[0]);
}
float2 __attribute__((overloadable)) _convert_float2(fp4e2m1_t2 val) {
    return (float2)(_intel_convert_fp4_to_f16(val.data[0]), _intel_convert_fp4_to_f16(val.data[1]));
}
float3 __attribute__((overloadable)) _convert_float3(fp4e2m1_t3 val) {
    return (float3)(_intel_convert_fp4_to_f16(val.data[0]), _intel_convert_fp4_to_f16(val.data[1]), _intel_convert_fp4_to_f16(val.data[2]));
}
float4 __attribute__((overloadable)) _convert_float4(fp4e2m1_t4 val) {
    return (float4)(_intel_convert_fp4_to_f16(val.data[0]), _intel_convert_fp4_to_f16(val.data[1]), 
                   _intel_convert_fp4_to_f16(val.data[2]), _intel_convert_fp4_to_f16(val.data[3]));
}
float8 __attribute__((overloadable)) _convert_float8(fp4e2m1_t8 val) {
    return (float8)(_intel_convert_fp4_to_f16(val.data[0]), _intel_convert_fp4_to_f16(val.data[1]),
                    _intel_convert_fp4_to_f16(val.data[2]), _intel_convert_fp4_to_f16(val.data[3]),
                    _intel_convert_fp4_to_f16(val.data[4]), _intel_convert_fp4_to_f16(val.data[5]),
                    _intel_convert_fp4_to_f16(val.data[6]), _intel_convert_fp4_to_f16(val.data[7]));
}
float16 __attribute__((overloadable)) _convert_float16(fp4e2m1_t16 val) {
    return (float16)(
        _intel_convert_fp4_to_f16(val.data[0]),  _intel_convert_fp4_to_f16(val.data[1]),
        _intel_convert_fp4_to_f16(val.data[2]),  _intel_convert_fp4_to_f16(val.data[3]),
        _intel_convert_fp4_to_f16(val.data[4]),  _intel_convert_fp4_to_f16(val.data[5]),
        _intel_convert_fp4_to_f16(val.data[6]),  _intel_convert_fp4_to_f16(val.data[7]),
        _intel_convert_fp4_to_f16(val.data[8]),  _intel_convert_fp4_to_f16(val.data[9]),
        _intel_convert_fp4_to_f16(val.data[10]), _intel_convert_fp4_to_f16(val.data[11]),
        _intel_convert_fp4_to_f16(val.data[12]), _intel_convert_fp4_to_f16(val.data[13]),
        _intel_convert_fp4_to_f16(val.data[14]), _intel_convert_fp4_to_f16(val.data[15])
    );
}




fp4e2m1_t __attribute__((overloadable)) _convert_fp4e2m1_t(float val) {
    fp4e2m1_t res;
    res.data = _intel_convert_f16_to_f4((half)val);
    return res;
}
fp4e2m1_t1 __attribute__((overloadable)) _convert_fp4e2m1_t1(float val[1]) {
    fp4e2m1_t1 res;
    res.data[0] = _intel_convert_f16_to_f4((half)val[0]);
    return res;
}
fp4e2m1_t2 __attribute__((overloadable)) _convert_fp4e2m1_t2(float2 val) {
    fp4e2m1_t2 res;
    res.data[0] = _intel_convert_f16_to_f4((half)val.x);
    res.data[1] = _intel_convert_f16_to_f4((half)val.y);
    return res;
}
fp4e2m1_t3 __attribute__((overloadable)) _convert_fp4e2m1_t3(float3 val) {
    fp4e2m1_t3 res;
    res.data[0] = _intel_convert_f16_to_f4((half)val.x);
    res.data[1] = _intel_convert_f16_to_f4((half)val.y);
    res.data[2] = _intel_convert_f16_to_f4((half)val.z);
    return res;
}
fp4e2m1_t4 __attribute__((overloadable)) _convert_fp4e2m1_t4(float4 val) {
    fp4e2m1_t4 res;
    res.data[0] = _intel_convert_f16_to_f4((half)val.x);
    res.data[1] = _intel_convert_f16_to_f4((half)val.y);
    res.data[2] = _intel_convert_f16_to_f4((half)val.z);
    res.data[3] = _intel_convert_f16_to_f4((half)val.w);
    return res;
}
fp4e2m1_t8 __attribute__((overloadable)) _convert_fp4e2m1_t8(float8 val) {
    fp4e2m1_t8 res;
    res.data[0] = _intel_convert_f16_to_f4((half)val.s0);
    res.data[1] = _intel_convert_f16_to_f4((half)val.s1);
    res.data[2] = _intel_convert_f16_to_f4((half)val.s2);
    res.data[3] = _intel_convert_f16_to_f4((half)val.s3);
    res.data[4] = _intel_convert_f16_to_f4((half)val.s4);
    res.data[5] = _intel_convert_f16_to_f4((half)val.s5);
    res.data[6] = _intel_convert_f16_to_f4((half)val.s6);
    res.data[7] = _intel_convert_f16_to_f4((half)val.s7);
    return res;
}
fp4e2m1_t16 __attribute__((overloadable)) _convert_fp4e2m1_t16(float16 val) {
    fp4e2m1_t16 res;
    res.data[0]  = _intel_convert_f16_to_f4((half)val.s0);
    res.data[1]  = _intel_convert_f16_to_f4((half)val.s1);
    res.data[2]  = _intel_convert_f16_to_f4((half)val.s2);
    res.data[3]  = _intel_convert_f16_to_f4((half)val.s3);
    res.data[4]  = _intel_convert_f16_to_f4((half)val.s4);
    res.data[5]  = _intel_convert_f16_to_f4((half)val.s5);
    res.data[6]  = _intel_convert_f16_to_f4((half)val.s6);
    res.data[7]  = _intel_convert_f16_to_f4((half)val.s7);
    res.data[8]  = _intel_convert_f16_to_f4((half)val.s8);
    res.data[9]  = _intel_convert_f16_to_f4((half)val.s9);
    res.data[10] = _intel_convert_f16_to_f4((half)val.sA);
    res.data[11] = _intel_convert_f16_to_f4((half)val.sB);
    res.data[12] = _intel_convert_f16_to_f4((half)val.sC);
    res.data[13] = _intel_convert_f16_to_f4((half)val.sD);
    res.data[14] = _intel_convert_f16_to_f4((half)val.sE);
    res.data[15] = _intel_convert_f16_to_f4((half)val.sF);
    return res;
}

fp4e2m1_t __attribute__((overloadable)) _convert_fp4e2m1_t(half val) {
    fp4e2m1_t res;
    res.data = _intel_convert_f16_to_f4(val);
    return res;
}
fp4e2m1_t1 __attribute__((overloadable)) _convert_fp4e2m1_t1(half val[1]) {
    fp4e2m1_t1 res;
    res.data[0] = _intel_convert_f16_to_f4(val[0]);
    return res;
}
fp4e2m1_t2 __attribute__((overloadable)) _convert_fp4e2m1_t2(half2 val) {
    fp4e2m1_t2 res;
    res.data[0] = _intel_convert_f16_to_f4(val.x);
    res.data[1] = _intel_convert_f16_to_f4(val.y);
    return res;
}
fp4e2m1_t3 __attribute__((overloadable)) _convert_fp4e2m1_t3(half3 val) {
    fp4e2m1_t3 res;
    res.data[0] = _intel_convert_f16_to_f4(val.x);
    res.data[1] = _intel_convert_f16_to_f4(val.y);
    res.data[2] = _intel_convert_f16_to_f4(val.z);
    return res;
}
fp4e2m1_t4 __attribute__((overloadable)) _convert_fp4e2m1_t4(half4 val) {
    fp4e2m1_t4 res;
    res.data[0] = _intel_convert_f16_to_f4(val.x);
    res.data[1] = _intel_convert_f16_to_f4(val.y);
    res.data[2] = _intel_convert_f16_to_f4(val.z);
    res.data[3] = _intel_convert_f16_to_f4(val.w);
    return res;
}
fp4e2m1_t8 __attribute__((overloadable)) _convert_fp4e2m1_t8(half8 val) {
    fp4e2m1_t8 res;
    res.data[0] = _intel_convert_f16_to_f4(val.s0);
    res.data[1] = _intel_convert_f16_to_f4(val.s1);
    res.data[2] = _intel_convert_f16_to_f4(val.s2);
    res.data[3] = _intel_convert_f16_to_f4(val.s3);
    res.data[4] = _intel_convert_f16_to_f4(val.s4);
    res.data[5] = _intel_convert_f16_to_f4(val.s5);
    res.data[6] = _intel_convert_f16_to_f4(val.s6);
    res.data[7] = _intel_convert_f16_to_f4(val.s7);
    return res;
}
fp4e2m1_t16 __attribute__((overloadable)) _convert_fp4e2m1_t16(half16 val) {
    fp4e2m1_t16 res;
    res.data[0]  = _intel_convert_f16_to_f4(val.s0);
    res.data[1]  = _intel_convert_f16_to_f4(val.s1);
    res.data[2]  = _intel_convert_f16_to_f4(val.s2);
    res.data[3]  = _intel_convert_f16_to_f4(val.s3);
    res.data[4]  = _intel_convert_f16_to_f4(val.s4);
    res.data[5]  = _intel_convert_f16_to_f4(val.s5);
    res.data[6]  = _intel_convert_f16_to_f4(val.s6);
    res.data[7]  = _intel_convert_f16_to_f4(val.s7);
    res.data[8]  = _intel_convert_f16_to_f4(val.s8);
    res.data[9]  = _intel_convert_f16_to_f4(val.s9);
    res.data[10] = _intel_convert_f16_to_f4(val.sA);
    res.data[11] = _intel_convert_f16_to_f4(val.sB);
    res.data[12] = _intel_convert_f16_to_f4(val.sC);
    res.data[13] = _intel_convert_f16_to_f4(val.sD);
    res.data[14] = _intel_convert_f16_to_f4(val.sE);
    res.data[15] = _intel_convert_f16_to_f4(val.sF);
    return res;
}

fp4e2m1_t __attribute__((overloadable)) _convert_fp4e2m1_t_sat(float val) {
    fp4e2m1_t res;
    res.data = _intel_convert_f16_to_f4_sat((half)val);
    return res;
}
fp4e2m1_t1 __attribute__((overloadable)) _convert_fp4e2m1_t1_sat(float val[1]) {
    fp4e2m1_t1 res;
    res.data[0] = _intel_convert_f16_to_f4_sat((half)val[0]);
    return res;
}
fp4e2m1_t2 __attribute__((overloadable)) _convert_fp4e2m1_t2_sat(float2 val) {
    fp4e2m1_t2 res;
    res.data[0] = _intel_convert_f16_to_f4_sat((half)val.x);
    res.data[1] = _intel_convert_f16_to_f4_sat((half)val.y);
    return res;
}
fp4e2m1_t3 __attribute__((overloadable)) _convert_fp4e2m1_t3_sat(float3 val) {
    fp4e2m1_t3 res;
    res.data[0] = _intel_convert_f16_to_f4_sat((half)val.x);
    res.data[1] = _intel_convert_f16_to_f4_sat((half)val.y);
    res.data[2] = _intel_convert_f16_to_f4_sat((half)val.z);
    return res;
}
fp4e2m1_t4 __attribute__((overloadable)) _convert_fp4e2m1_t4_sat(float4 val) {
    fp4e2m1_t4 res;
    res.data[0] = _intel_convert_f16_to_f4_sat((half)val.x);
    res.data[1] = _intel_convert_f16_to_f4_sat((half)val.y);
    res.data[2] = _intel_convert_f16_to_f4_sat((half)val.z);
    res.data[3] = _intel_convert_f16_to_f4_sat((half)val.w);
    return res;
}
fp4e2m1_t8 __attribute__((overloadable)) _convert_fp4e2m1_t8_sat(float8 val) {
    fp4e2m1_t8 res;
    res.data[0] = _intel_convert_f16_to_f4_sat((half)val.s0);
    res.data[1] = _intel_convert_f16_to_f4_sat((half)val.s1);
    res.data[2] = _intel_convert_f16_to_f4_sat((half)val.s2);
    res.data[3] = _intel_convert_f16_to_f4_sat((half)val.s3);
    res.data[4] = _intel_convert_f16_to_f4_sat((half)val.s4);
    res.data[5] = _intel_convert_f16_to_f4_sat((half)val.s5);
    res.data[6] = _intel_convert_f16_to_f4_sat((half)val.s6);
    res.data[7] = _intel_convert_f16_to_f4_sat((half)val.s7);
    return res;
}
fp4e2m1_t16 __attribute__((overloadable)) _convert_fp4e2m1_t16_sat(float16 val) {
    fp4e2m1_t16 res;
    res.data[0]  = _intel_convert_f16_to_f4_sat((half)val.s0);
    res.data[1]  = _intel_convert_f16_to_f4_sat((half)val.s1);
    res.data[2]  = _intel_convert_f16_to_f4_sat((half)val.s2);
    res.data[3]  = _intel_convert_f16_to_f4_sat((half)val.s3);
    res.data[4]  = _intel_convert_f16_to_f4_sat((half)val.s4);
    res.data[5]  = _intel_convert_f16_to_f4_sat((half)val.s5);
    res.data[6]  = _intel_convert_f16_to_f4_sat((half)val.s6);
    res.data[7]  = _intel_convert_f16_to_f4_sat((half)val.s7);
    res.data[8]  = _intel_convert_f16_to_f4_sat((half)val.s8);
    res.data[9]  = _intel_convert_f16_to_f4_sat((half)val.s9);
    res.data[10] = _intel_convert_f16_to_f4_sat((half)val.sA);
    res.data[11] = _intel_convert_f16_to_f4_sat((half)val.sB);
    res.data[12] = _intel_convert_f16_to_f4_sat((half)val.sC);
    res.data[13] = _intel_convert_f16_to_f4_sat((half)val.sD);
    res.data[14] = _intel_convert_f16_to_f4_sat((half)val.sE);
    res.data[15] = _intel_convert_f16_to_f4_sat((half)val.sF);
    return res;
}

fp4e2m1_t __attribute__((overloadable)) _convert_fp4e2m1_t_sat(half val) {
    fp4e2m1_t res;
    res.data = _intel_convert_f16_to_f4_sat(val);
    return res;
}
fp4e2m1_t1 __attribute__((overloadable)) _convert_fp4e2m1_t1_sat(half val[1]) {
    fp4e2m1_t1 res;
    res.data[0] = _intel_convert_f16_to_f4_sat(val[0]);
    return res;
}
fp4e2m1_t2 __attribute__((overloadable)) _convert_fp4e2m1_t2_sat(half2 val) {
    fp4e2m1_t2 res;
    res.data[0] = _intel_convert_f16_to_f4_sat(val.x);
    res.data[1] = _intel_convert_f16_to_f4_sat(val.y);
    return res;
}
fp4e2m1_t3 __attribute__((overloadable)) _convert_fp4e2m1_t3_sat(half3 val) {
    fp4e2m1_t3 res;
    res.data[0] = _intel_convert_f16_to_f4_sat(val.x);
    res.data[1] = _intel_convert_f16_to_f4_sat(val.y);
    res.data[2] = _intel_convert_f16_to_f4_sat(val.z);
    return res;
}
fp4e2m1_t4 __attribute__((overloadable)) _convert_fp4e2m1_t4_sat(half4 val) {
    fp4e2m1_t4 res;
    res.data[0] = _intel_convert_f16_to_f4_sat(val.x);
    res.data[1] = _intel_convert_f16_to_f4_sat(val.y);
    res.data[2] = _intel_convert_f16_to_f4_sat(val.z);
    res.data[3] = _intel_convert_f16_to_f4_sat(val.w);
    return res;
}
fp4e2m1_t8 __attribute__((overloadable)) _convert_fp4e2m1_t8_sat(half8 val) {
    fp4e2m1_t8 res;
    res.data[0] = _intel_convert_f16_to_f4_sat(val.s0);
    res.data[1] = _intel_convert_f16_to_f4_sat(val.s1);
    res.data[2] = _intel_convert_f16_to_f4_sat(val.s2);
    res.data[3] = _intel_convert_f16_to_f4_sat(val.s3);
    res.data[4] = _intel_convert_f16_to_f4_sat(val.s4);
    res.data[5] = _intel_convert_f16_to_f4_sat(val.s5);
    res.data[6] = _intel_convert_f16_to_f4_sat(val.s6);
    res.data[7] = _intel_convert_f16_to_f4_sat(val.s7);
    return res;
}
fp4e2m1_t16 __attribute__((overloadable)) _convert_fp4e2m1_t16_sat(half16 val) {
    fp4e2m1_t16 res;
    res.data[0]  = _intel_convert_f16_to_f4_sat(val.s0);
    res.data[1]  = _intel_convert_f16_to_f4_sat(val.s1);
    res.data[2]  = _intel_convert_f16_to_f4_sat(val.s2);
    res.data[3]  = _intel_convert_f16_to_f4_sat(val.s3);
    res.data[4]  = _intel_convert_f16_to_f4_sat(val.s4);
    res.data[5]  = _intel_convert_f16_to_f4_sat(val.s5);
    res.data[6]  = _intel_convert_f16_to_f4_sat(val.s6);
    res.data[7]  = _intel_convert_f16_to_f4_sat(val.s7);
    res.data[8]  = _intel_convert_f16_to_f4_sat(val.s8);
    res.data[9]  = _intel_convert_f16_to_f4_sat(val.s9);
    res.data[10] = _intel_convert_f16_to_f4_sat(val.sA);
    res.data[11] = _intel_convert_f16_to_f4_sat(val.sB);
    res.data[12] = _intel_convert_f16_to_f4_sat(val.sC);
    res.data[13] = _intel_convert_f16_to_f4_sat(val.sD);
    res.data[14] = _intel_convert_f16_to_f4_sat(val.sE);
    res.data[15] = _intel_convert_f16_to_f4_sat(val.sF);
    return res;
}

fp4e2m1_t __attribute__((overloadable)) as_fp4e2m1_t(uchar val) {
    fp4e2m1_t res;
    res.data = val;
    return res;
}
fp4e2m1_t1 __attribute__((overloadable)) as_fp4e2m1_t1(uchar val[1]) {
    fp4e2m1_t1 res;
    res.data[0] = val[0];
    return res;
}
fp4e2m1_t2 __attribute__((overloadable)) as_fp4e2m1_t2(uchar2 val) {
    fp4e2m1_t2 res;
    res.data[0] = val.x;
    res.data[1] = val.y;
    return res;
}
fp4e2m1_t3 __attribute__((overloadable)) as_fp4e2m1_t3(uchar3 val) {
    fp4e2m1_t3 res;
    res.data[0] = val.x;
    res.data[1] = val.y;
    res.data[2] = val.z;
    return res;
}
fp4e2m1_t4 __attribute__((overloadable)) as_fp4e2m1_t4(uchar4 val) {
    fp4e2m1_t4 res;
    res.data[0] = val.x;
    res.data[1] = val.y;
    res.data[2] = val.z;
    res.data[3] = val.w;
    return res;
}
fp4e2m1_t8 __attribute__((overloadable)) as_fp4e2m1_t8(uchar8 val) {
    fp4e2m1_t8 res;
    res.data[0] = val.s0;
    res.data[1] = val.s1;
    res.data[2] = val.s2;
    res.data[3] = val.s3;
    res.data[4] = val.s4;
    res.data[5] = val.s5;
    res.data[6] = val.s6;
    res.data[7] = val.s7;
    return res;
}
fp4e2m1_t16 __attribute__((overloadable)) as_fp4e2m1_t16(uchar16 val) {
    fp4e2m1_t16 res;
    res.data[0]  = val.s0;
    res.data[1]  = val.s1;
    res.data[2]  = val.s2;
    res.data[3]  = val.s3;
    res.data[4]  = val.s4;
    res.data[5]  = val.s5;
    res.data[6]  = val.s6;
    res.data[7]  = val.s7;
    res.data[8]  = val.s8;
    res.data[9]  = val.s9;
    res.data[10] = val.sA;
    res.data[11] = val.sB;
    res.data[12] = val.sC;
    res.data[13] = val.sD;
    res.data[14] = val.sE;
    res.data[15] = val.sF;
    return res;
}



uchar __attribute__((overloadable)) _as_uchar(fp4e2m1_t val) {
    return val.data;
}
uchar __attribute__((overloadable)) _as_uchar(fp4e2m1_t1 val) {
    return val.data[0];
}
uchar2 __attribute__((overloadable)) _as_uchar2(fp4e2m1_t2 val) {
    return (uchar2)(val.data[0], val.data[1]);
}
uchar3 __attribute__((overloadable)) _as_uchar3(fp4e2m1_t3 val) {
    return (uchar3)(val.data[0], val.data[1], val.data[2]);
}
uchar4 __attribute__((overloadable)) _as_uchar4(fp4e2m1_t4 val) {
    return (uchar4)(val.data[0], val.data[1], val.data[2], val.data[3]);
}
uchar8 __attribute__((overloadable)) _as_uchar8(fp4e2m1_t8 val) {
    return (uchar8)(val.data[0], val.data[1], val.data[2], val.data[3],
                    val.data[4], val.data[5], val.data[6], val.data[7]);
}
uchar16 __attribute__((overloadable)) _as_uchar16(fp4e2m1_t16 val) {
    return (uchar16)(val.data[0],  val.data[1],  val.data[2],  val.data[3],
                     val.data[4],  val.data[5],  val.data[6],  val.data[7],
                     val.data[8],  val.data[9],  val.data[10], val.data[11],
                     val.data[12], val.data[13], val.data[14], val.data[15]);
}