/*******************************************************************************
* Copyright 2019 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

// #include "ocl_math_utils.h"

#define for_ for

#define CONCAt2(a, b) a##b
#define CONCAT2(a, b) CONCAt2(a, b)

#if DT_F32 == 1
#define DATA_T float
#define DATA8_T float8
#define DATA_MAX FLT_MAX
#define DATA_MIN -DATA_MAX
#define DATA_ZERO 0.0f
#define DATA_ONE 1.0f
#define DEF_ACC_DATA_T float
#define DEF_ACC_DATA8_T float8
#define POST_OP_DATA_T float
#define TO_DATA_T(v) static_cast<float>(v)
#define TO_DEF_ACC_DATA_T(v) static_cast<float>(v)
#define DATA_TO_REF convert_float
#define CONVERT_DATA_T convert_float
#define CONVERT_DATA8_T convert_float8
#define CONVERT_FLOAT_T convert_float
#define CONVERT_FLOAT8_T convert_float8
#define ROUND

#define BLOCK_READ intel_sub_group_block_read
#define BLOCK_WRITE intel_sub_group_block_write
#define BLOCK_READ8 intel_sub_group_block_read8
#define BLOCK_WRITE8 intel_sub_group_block_write8

#define AS_DATA_T as_float
#define AS_DATA8_T as_float8

#define AS_UINT_T as_uint
#define AS_UINT8_T as_uint8

#define BLOCK_DATA_T uint
#define BLOCK_DATA8_T uint8
#define AS_BLOCK_DATA_T as_uint
#define AS_BLOCK_DATA8_T as_uint8
#elif DT_F16 == 1
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define DATA_T half
#define DATA8_T half8
#define DATA_MAX HALF_MAX
#define DATA_MIN -DATA_MAX
#define DATA_ZERO 0.0h
#define DATA_ONE 1.0h
#define DEF_ACC_DATA_T half
#define DEF_ACC_DATA8_T half8
#define POST_OP_DATA_T half
#define TO_DATA_T(v) (half)(v)
#define TO_DEF_ACC_DATA_T(v) (half)(v)
#define DATA_TO_REF convert_half
#define CONVERT_DATA_T convert_half
#define CONVERT_DATA8_T convert_half8
#define CONVERT_FLOAT_T convert_float
#define CONVERT_FLOAT8_T convert_float8
#define ROUND

#define BLOCK_READ intel_sub_group_block_read_us
#define BLOCK_WRITE intel_sub_group_block_write_us
#define BLOCK_READ8 intel_sub_group_block_read_us8
#define BLOCK_WRITE8 intel_sub_group_block_write_us8
#define AS_DATA_T as_half
#define AS_DATA8_T as_half8

#define AS_UINT_T as_ushort
#define AS_UINT8_T as_ushort8

#define BLOCK_DATA_T ushort
#define BLOCK_DATA8_T ushort8
#define AS_BLOCK_DATA_T as_ushort
#define AS_BLOCK_DATA8_T as_ushort8
#elif DT_BF16 == 1
#define DATA_T ushort
#define POST_OP_DATA_T float
#define DATA8_T ushort8
#define DATA_MAX 3.38953138925153547590470800371487866880e+38F
#define DATA_MIN (-DATA_MAX)
#define DATA_ZERO 0.0f
#define DATA_ONE 1.0f
#define DEF_ACC_DATA_T float
#define DEF_ACC_DATA8_T float8
#define TO_DATA_T(v) convert_f32_to_bf16(v)
#define TO_DEF_ACC_DATA_T(v) convert_bf16_to_f32(v)
#define DATA_TO_REF convert_bf16_to_f32
#define CONVERT_DATA_T convert_f32_to_bf16
#define CONVERT_DATA8_T convert_f32_to_bf16_vec8
#define CONVERT_FLOAT_T convert_bf16_to_f32
#define CONVERT_FLOAT8_T convert_bf16_to_f32_vec8
#define ROUND

#define BLOCK_READ intel_sub_group_block_read_us
#define BLOCK_WRITE intel_sub_group_block_write_us
#define BLOCK_READ8 intel_sub_group_block_read_us8
#define BLOCK_WRITE8 intel_sub_group_block_write_us8
#define AS_DATA_T as_ushort
#define AS_DATA8_T as_ushort8

#define AS_UINT_T as_ushort
#define AS_UINT8_T as_ushort8

#define BLOCK_DATA_T ushort
#define BLOCK_DATA8_T ushort8
#define AS_BLOCK_DATA_T as_ushort
#define AS_BLOCK_DATA8_T as_ushort8
#elif DT_S8 == 1
#define DATA_T char
#define DATA8_T char8
#define DATA_MAX CHAR_MAX
#define DATA_MIN CHAR_MIN
#define DATA_ZERO 0
#define DATA_ONE 1
#define DEF_ACC_DATA_T int
#define DEF_ACC_DATA8_T int8
#define POST_OP_DATA_T float
#define TO_DATA_T(v) static_cast<char>(v)
#define DATA_TO_REF convert_char
#define CONVERT_DATA_T convert_char
#define CONVERT_DATA8_T convert_char8
#define ROUND rint

#define BLOCK_READ intel_sub_group_block_read_uc
#define BLOCK_WRITE intel_sub_group_block_write_uc
#define BLOCK_READ8 intel_sub_group_block_read_uc8
#define BLOCK_WRITE8 intel_sub_group_block_write_uc8
#define AS_DATA_T as_char
#define AS_DATA8_T as_char8

#define AS_UINT_T as_uchar
#define AS_UINT8_T as_uchar8

#define BLOCK_DATA_T uchar
#define BLOCK_DATA8_T uchar8
#define AS_BLOCK_DATA_T as_uchar
#define AS_BLOCK_DATA8_T as_uchar8
#elif DT_U8 == 1
#define DATA_T uchar
#define DATA8_T uchar8
#define DATA_MAX UCHAR_MAX
#define DATA_MIN 0
#define DATA_ZERO 0
#define DATA_ONE 1
#define DEF_ACC_DATA_T int
#define DEF_ACC_DATA8_T int8
#define POST_OP_DATA_T float
#define TO_DATA_T(v) (uchar)(v)
#define DATA_TO_REF convert_uchar
#define CONVERT_DATA_T convert_uchar
#define CONVERT_DATA8_T convert_uchar8
#define ROUND rint

#define BLOCK_READ intel_sub_group_block_read_uc
#define BLOCK_WRITE intel_sub_group_block_write_uc
#define BLOCK_READ8 intel_sub_group_block_read_uc8
#define BLOCK_WRITE8 intel_sub_group_block_write_uc8
#define AS_DATA_T as_uchar
#define AS_DATA8_T as_uchar8

#define AS_UINT_T as_uchar
#define AS_UINT8_T as_uchar8

#define BLOCK_DATA_T uchar
#define BLOCK_DATA8_T uchar8
#define AS_BLOCK_DATA_T as_uchar
#define AS_BLOCK_DATA8_T as_uchar8
#elif DT_S32 == 1
#define DATA_T int
#define CONVERT_DATA_T convert_int_sat_rte
#define POST_OP_DATA_T float
#elif !defined(DT_UNDEF)
#error "Unexpected data type"
#endif

#if VECT_DT_N == 1
#define VECT_DATA_T DATA_T
#define VECT_DEF_ACC_DATA_T DEF_ACC_DATA_T
#define AS_VECT_DATA_T AS_DATA_T
#define VECT_BLOCK_READ BLOCK_READ
#define VECT_BLOCK_WRITE BLOCK_WRITE
#define VECT_UINT_READ intel_sub_group_block_read
#define VECT_UINT_WRITE intel_sub_group_block_write
#define VECT_BLOCK_DATA_T BLOCK_DATA_T
#define AS_VECT_BLOCK_DATA_T AS_BLOCK_DATA_T
#define CONVERT_VECT_FLOAT_T CONVERT_FLOAT_T
#define CONVERT_VECTOR_DATA_T CONVERT_DATA_T
#define VECT_INT_T int
#define VECT_UINT_T uint
#define VECT_FLOAT_T float
#define AS_VECT_INT_T as_int
#define AS_VECT_UINT_T as_uint
#elif VECT_DT_N == 8
#define VECT_DATA_T DATA8_T
#define VECT_DEF_ACC_DATA_T DEF_ACC_DATA8_T
#define AS_VECT_DATA_T AS_DATA8_T
#define VECT_BLOCK_READ BLOCK_READ8
#define VECT_BLOCK_WRITE BLOCK_WRITE8
#define VECT_UINT_READ intel_sub_group_block_read8
#define VECT_UINT_WRITE intel_sub_group_block_write8
#define VECT_BLOCK_DATA_T BLOCK_DATA8_T
#define AS_VECT_BLOCK_DATA_T AS_BLOCK_DATA8_T
#define CONVERT_VECT_FLOAT_T CONVERT_FLOAT8_T
#define CONVERT_VECTOR_DATA_T CONVERT_DATA8_T
#define VECT_INT_T int8
#define VECT_UINT_T uint8
#define VECT_FLOAT_T float8
#define AS_VECT_INT_T as_int8
#define AS_VECT_UINT_T as_uint8
#endif

#ifdef SRC_DATA_T
#define SRC_DATA8_T CONCAT2(SRC_DATA_T, 8)
#if SRC_DT_BF16
#define SRC_TO_REF(x) convert_bf16_to_f32(x)
#define SRC_TO_REF8(x) convert_bf16_to_f32_vec8(x)
#else
#define SRC_TO_REF(x) (x)
#define SRC_TO_REF8(x) (x)
#endif
#if SRC_DT_BF16
#define TO_SRC(x) convert_f32_to_bf16(x)
#elif SRC_DT_U8
#define TO_SRC(x) convert_uchar_sat_rte(x)
#elif SRC_DT_S8
#define TO_SRC(x) convert_char_sat_rte(x)
#elif SRC_DT_S32
#define TO_SRC(x) convert_int_sat_rte(x)
#else
#define TO_SRC(x) (x)
#endif
#endif

#ifdef WEI_DATA_T
#if WEI_DT_BF16
#define WEI_TO_REF(x) convert_bf16_to_f32(x)
#define REF_TO_WEI(x) convert_f32_to_bf16(x)
#else
#define WEI_TO_REF(x) (x)
#define REF_TO_WEI(x) (x)
#endif
#if WEI_DT_BF16
#define TO_WEI(x) convert_f32_to_bf16(x)
#elif WEI_DT_U8
#define TO_WEI(x) convert_uchar_sat_rte(x)
#elif WEI_DT_S8
#define TO_WEI(x) convert_char_sat_rte(x)
#elif WEI_DT_S32
#define TO_WEI(x) convert_int_sat_rte(x)
#else
#define TO_WEI(x) (x)
#endif
#endif

#ifdef BIA_DATA_T
#if BIA_DT_BF16
#define BIA_TO_REF(x) convert_bf16_to_f32(x)
#define REF_TO_BIA(x) convert_f32_to_bf16(x)
#else
#define BIA_TO_REF(x) (x)
#define REF_TO_BIA(x) (x)
#endif
#if BIA_DT_BF16
#define TO_BIA(x) convert_f32_to_bf16(x)
#elif BIA_DT_U8
#define TO_BIA(x) convert_uchar_sat_rte(x)
#elif BIA_DT_S8
#define TO_BIA(x) convert_char_sat_rte(x)
#elif BIA_DT_S32
#define TO_BIA(x) convert_int_sat_rte(x)
#else
#define TO_BIA(x) (x)
#endif
#endif

#ifdef DST_DATA_T
#define DST_DATA8_T CONCAT2(DST_DATA_T, 8)
#if DST_DT_BF16
#define DST_TO_REF(x) convert_bf16_to_f32(x)
#define DST_TO_REF8(x) convert_bf16_to_f32_vec8(x)
#define REF_TO_DST(x) convert_f32_to_bf16(x)
#define REF_TO_DST8(x) convert_f32_to_bf16_vec8(convert_float8(x))
#else
#define DST_TO_REF(x) (x)
#define DST_TO_REF8(x) (x)
#define REF_TO_DST(x) (x)
#define REF_TO_DST8(x) (x)
#endif
#if DST_DT_BF16
#define TO_DST(x) convert_f32_to_bf16(x)
#define TO_DST8(x) convert_f32_to_bf16_vec8(convert_float8(x))
#elif DST_DT_F16
#define TO_DST(x) convert_half(x)
#define TO_DST8(x) convert_half8(x)
#elif DST_DT_U8
#define TO_DST(x) convert_uchar_sat_rte(x)
#define TO_DST8(x) convert_uchar8_sat_rte(x)
#elif DST_DT_S8
#define TO_DST(x) convert_char_sat_rte(x)
#define TO_DST8(x) convert_char8_sat_rte(x)
#elif DST_DT_S32
#define TO_DST(x) convert_int_sat_rte(x)
#define TO_DST8(x) convert_int8_sat_rte(x)
#elif DST_DT_F32
#define TO_DST(x) convert_float(x)
#define TO_DST8(x) convert_float8(x)
#else
#error "Not expected"
#endif
#endif

#ifdef ACC_DATA_T
#if ACC_DT_F16
#define TO_ACC(x) convert_half(x)
#elif ACC_DT_F32
#define TO_ACC(x) convert_float(x)
#elif ACC_DT_S32
#define TO_ACC(x) convert_int(x)
#else
#error "Unexpected accumulation data type"
#endif
#endif

#define OFF_MD(prefix, x0, x1, x2, x3, x4, x5) \
    ((x0 / prefix##_B0_2) / prefix##_B0_1 * prefix##_S0_0) \
            + ((x0 / prefix##_B0_2) % prefix##_B0_1 * prefix##_S0_1) \
            + ((x0 % prefix##_B0_2) * prefix##_S0_2) \
            + ((x1 / prefix##_B1_2) / prefix##_B1_1 * prefix##_S1_0) \
            + ((x1 / prefix##_B1_2) % prefix##_B1_1 * prefix##_S1_1) \
            + ((x1 % prefix##_B1_2) * prefix##_S1_2) \
            + ((x2 / prefix##_B2_2) / prefix##_B2_1 * prefix##_S2_0) \
            + ((x2 / prefix##_B2_2) % prefix##_B2_1 * prefix##_S2_1) \
            + ((x2 % prefix##_B2_2) * prefix##_S2_2) \
            + ((x3 / prefix##_B3_2) / prefix##_B3_1 * prefix##_S3_0) \
            + ((x3 / prefix##_B3_2) % prefix##_B3_1 * prefix##_S3_1) \
            + ((x3 % prefix##_B3_2) * prefix##_S3_2) \
            + ((x4 / prefix##_B4_2) / prefix##_B4_1 * prefix##_S4_0) \
            + ((x4 / prefix##_B4_2) % prefix##_B4_1 * prefix##_S4_1) \
            + ((x4 % prefix##_B4_2) * prefix##_S4_2) \
            + ((x5 / prefix##_B5_2) / prefix##_B5_1 * prefix##_S5_0) \
            + ((x5 / prefix##_B5_2) % prefix##_B5_1 * prefix##_S5_1) \
            + ((x5 % prefix##_B5_2) * prefix##_S5_2)

#if NDIMS == 3
#define SRC_OFF(x0, x1, d, h, x2) \
    (((x0) % SRC_B0) * SRC_SB0 + ((x0) / SRC_B0) * SRC_S0 \
            + ((x1) % SRC_B1) * SRC_SB1 + ((x1) / SRC_B1) * SRC_S1 \
            + ((x2) % SRC_B2) * SRC_SB2 + ((x2) / SRC_B2) * SRC_S2)

#if WITH_GROUPS == 1
#define WHT_OFF(x0, x1, x2, d, h, x3) \
    (((x0) % WHT_B0) * WHT_SB0 + ((x0) / WHT_B0) * WHT_S0 \
            + ((x1) % WHT_B1) * WHT_SB1 + ((x1) / WHT_B1) * WHT_S1 \
            + ((x2) % WHT_B2) * WHT_SB2 + ((x2) / WHT_B2) * WHT_S2 \
            + ((x3) % WHT_B3) * WHT_SB3 + ((x3) / WHT_B3) * WHT_S3)
#else
#define WHT_OFF(g, x0, x1, d, h, x2) \
    (((x0) % WHT_B0) * WHT_SB0 + ((x0) / WHT_B0) * WHT_S0 \
            + ((x1) % WHT_B1) * WHT_SB1 + ((x1) / WHT_B1) * WHT_S1 \
            + ((x2) % WHT_B2) * WHT_SB2 + ((x2) / WHT_B2) * WHT_S2)
#endif

#define DST_OFF(x0, x1, d, h, x2) \
    (((x0) % DST_B0) * DST_SB0 + ((x0) / DST_B0) * DST_S0 \
            + ((x1) % DST_B1) * DST_SB1 + ((x1) / DST_B1) * DST_S1 \
            + ((x2) % DST_B2) * DST_SB2 + ((x2) / DST_B2) * DST_S2)
#elif NDIMS == 4
#define SRC_OFF(x0, x1, d, x2, x3) \
    (((x0) % SRC_B0) * SRC_SB0 + ((x0) / SRC_B0) * SRC_S0 \
            + ((x1) % SRC_B1) * SRC_SB1 + ((x1) / SRC_B1) * SRC_S1 \
            + ((x2) % SRC_B2) * SRC_SB2 + ((x2) / SRC_B2) * SRC_S2 \
            + ((x3) % SRC_B3) * SRC_SB3 + ((x3) / SRC_B3) * SRC_S3)

#if WITH_GROUPS == 1
#define WHT_OFF(x0, x1, x2, d, x3, x4) \
    (((x0) % WHT_B0) * WHT_SB0 + ((x0) / WHT_B0) * WHT_S0 \
            + ((x1) % WHT_B1) * WHT_SB1 + ((x1) / WHT_B1) * WHT_S1 \
            + ((x2) % WHT_B2) * WHT_SB2 + ((x2) / WHT_B2) * WHT_S2 \
            + ((x3) % WHT_B3) * WHT_SB3 + ((x3) / WHT_B3) * WHT_S3 \
            + ((x4) % WHT_B4) * WHT_SB4 + ((x4) / WHT_B4) * WHT_S4)
#else
#define WHT_OFF(g, x1, x2, d, x3, x4) \
    (((x1) % WHT_B0) * WHT_SB0 + ((x1) / WHT_B0) * WHT_S0 \
            + ((x2) % WHT_B1) * WHT_SB1 + ((x2) / WHT_B1) * WHT_S1 \
            + ((x3) % WHT_B2) * WHT_SB2 + ((x3) / WHT_B2) * WHT_S2 \
            + ((x4) % WHT_B3) * WHT_SB3 + ((x4) / WHT_B3) * WHT_S3)
#endif

#define DST_OFF(x0, x1, d, x2, x3) \
    (((x0) % DST_B0) * DST_SB0 + ((x0) / DST_B0) * DST_S0 \
            + ((x1) % DST_B1) * DST_SB1 + ((x1) / DST_B1) * DST_S1 \
            + ((x2) % DST_B2) * DST_SB2 + ((x2) / DST_B2) * DST_S2 \
            + ((x3) % DST_B3) * DST_SB3 + ((x3) / DST_B3) * DST_S3)
#elif NDIMS == 5
#define SRC_OFF(x0, x1, x2, x3, x4) \
    (((x0) % SRC_B0) * SRC_SB0 + ((x0) / SRC_B0) * SRC_S0 \
            + ((x1) % SRC_B1) * SRC_SB1 + ((x1) / SRC_B1) * SRC_S1 \
            + ((x2) % SRC_B2) * SRC_SB2 + ((x2) / SRC_B2) * SRC_S2 \
            + ((x3) % SRC_B3) * SRC_SB3 + ((x3) / SRC_B3) * SRC_S3 \
            + ((x4) % SRC_B4) * SRC_SB4 + ((x4) / SRC_B4) * SRC_S4)

#if WITH_GROUPS == 1
#define WHT_OFF(x0, x1, x2, x3, x4, x5) \
    (((x0) % WHT_B0) * WHT_SB0 + ((x0) / WHT_B0) * WHT_S0 \
            + ((x1) % WHT_B1) * WHT_SB1 + ((x1) / WHT_B1) * WHT_S1 \
            + ((x2) % WHT_B2) * WHT_SB2 + ((x2) / WHT_B2) * WHT_S2 \
            + ((x3) % WHT_B3) * WHT_SB3 + ((x3) / WHT_B3) * WHT_S3 \
            + ((x4) % WHT_B4) * WHT_SB4 + ((x4) / WHT_B4) * WHT_S4 \
            + ((x5) % WHT_B5) * WHT_SB5 + ((x5) / WHT_B5) * WHT_S5)
#else
#define WHT_OFF(g, x1, x2, x3, x4, x5) \
    (((x1) % WHT_B0) * WHT_SB0 + ((x1) / WHT_B0) * WHT_S0 \
            + ((x2) % WHT_B1) * WHT_SB1 + ((x2) / WHT_B1) * WHT_S1 \
            + ((x3) % WHT_B2) * WHT_SB2 + ((x3) / WHT_B2) * WHT_S2 \
            + ((x4) % WHT_B3) * WHT_SB3 + ((x4) / WHT_B3) * WHT_S3 \
            + ((x5) % WHT_B4) * WHT_SB4 + ((x5) / WHT_B4) * WHT_S4)
#endif

#define DST_OFF(x0, x1, x2, x3, x4) \
    (((x0) % DST_B0) * DST_SB0 + ((x0) / DST_B0) * DST_S0 \
            + ((x1) % DST_B1) * DST_SB1 + ((x1) / DST_B1) * DST_S1 \
            + ((x2) % DST_B2) * DST_SB2 + ((x2) / DST_B2) * DST_S2 \
            + ((x3) % DST_B3) * DST_SB3 + ((x3) / DST_B3) * DST_S3 \
            + ((x4) % DST_B4) * DST_SB4 + ((x4) / DST_B4) * DST_S4)
#endif

