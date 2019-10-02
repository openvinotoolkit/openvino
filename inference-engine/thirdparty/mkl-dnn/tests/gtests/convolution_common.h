#if 0
/*******************************************************************************
* Copyright 2017-2019 Intel Corporation
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
#endif

#include "mkldnn.hpp"

#if defined(WITH_DW_CONV)
#define EXPAND_FORMATS(src, conv1_weights, conv1_bias, conv2_weights, conv2_bias, dst) \
    { mkldnn::memory::format::src, mkldnn::memory::format::conv1_weights, mkldnn::memory::format::conv1_bias, \
    mkldnn::memory::format::conv2_weights, mkldnn::memory::format::conv2_bias, mkldnn::memory::format::dst }
#elif defined(DEF)
#define EXPAND_FORMATS(src, offsets, weights, bias, dst) \
    { mkldnn::memory::format::src, mkldnn::memory::format::offsets, mkldnn::memory::format::weights, \
    mkldnn::memory::format::bias, mkldnn::memory::format::dst }
#else
#define EXPAND_FORMATS(src, weights, bias, dst) \
    { mkldnn::memory::format::src, mkldnn::memory::format::weights, \
    mkldnn::memory::format::bias, mkldnn::memory::format::dst }
#endif

#define EXPAND_ARGS(args) args

#define ENGINE mkldnn::engine::kind::cpu
#if defined(BIN)
#define ALGORITHM mkldnn::binary_convolution_direct
#elif defined(DEF)
#define ALGORITHM mkldnn::deformable_convolution_direct
#else
#define ALGORITHM mkldnn::convolution_direct
#endif

#ifdef DIRECTION_FORWARD
#if defined(FP32)
#define FMT_DATA_BLOCKED nChw8c
#define FMT_DATA_BLOCKED16 nChw16c
#define FMT_WEIGHTS_BLOCKED OIhw8i8o
#define FMT_WEIGHTS_BLOCKED_G gOIhw8i8o
#define FMT_WEIGHTS_BLOCKED16 OIhw16i16o
#define FMT_WEIGHTS_BLOCKED16_G gOIhw16i16o
#elif defined(S16S16S32)
#define FMT_DATA_BLOCKED nChw8c
#define FMT_DATA_BLOCKED16 nChw16c
#define FMT_WEIGHTS_BLOCKED OIhw8i8o
#define FMT_WEIGHTS_BLOCKED_G gOIhw8i8o
#define FMT_WEIGHTS_BLOCKED16 OIhw8i16o2i
#define FMT_WEIGHTS_BLOCKED16_G gOIhw8i16o2i
#elif defined(U8S8) || defined(S8S8)
#define FMT_DATA_BLOCKED nhwc
#define FMT_DATA_BLOCKED16 nhwc
#define FMT_DATA_BLOCKED_3D ndhwc
#define FMT_WEIGHTS_BLOCKED OhIw8o4i
#define FMT_WEIGHTS_BLOCKED_G gOhIw8o4i
#define FMT_WEIGHTS_BLOCKED16 OIhw4i16o4i
#define FMT_WEIGHTS_BLOCKED16_G gOIhw4i16o4i
#define FMT_WEIGHTS_BLOCKED_3D OdhIw8o4i
#define FMT_WEIGHTS_BLOCKED_G_3D gOdhIw8o4i
#define FMT_WEIGHTS_BLOCKED_SIGNED OhIw8o4i_s8s8
#define FMT_WEIGHTS_BLOCKED_G_SIGNED gOhIw8o4i_s8s8
#define FMT_WEIGHTS_BLOCKED_SIGNED_3D OdhIw8o4i_s8s8
#define FMT_WEIGHTS_BLOCKED_G_SIGNED_3D gOdhIw8o4i_s8s8
#elif defined(BIN)
#define FMT_DATA_BLOCKED nhwc
#define FMT_DATA_BLOCKED16 nhwc
#define FMT_WEIGHTS_BLOCKED OhIw8o32i
#define FMT_WEIGHTS_BLOCKED_G OhIw8o32i
#define FMT_WEIGHTS_BLOCKED16 OhIw16o32i
#define FMT_WEIGHTS_BLOCKED16_G OhIw16o32i
#define FMT_WEIGHTS_DW_BLOCKED Goihw8g
#define FMT_WEIGHTS_DW_BLOCKED16 Goihw16g
#endif
#define FMT_WEIGHTS_BLOCKED16_IOhw16o16i FMT_WEIGHTS_BLOCKED16
#define TEST_CASE_NAME_PREFIX Forward
#elif defined DIRECTION_BACKWARD_DATA
#define FMT_DATA_BLOCKED nChw8c
#define FMT_DATA_BLOCKED16 nChw16c
#define FMT_WEIGHTS_BLOCKED OIhw8o8i
#define FMT_WEIGHTS_BLOCKED_G gOIhw8o8i
#if defined(FP32)
#define FMT_WEIGHTS_BLOCKED16 OIhw16o16i
#define FMT_WEIGHTS_BLOCKED16_G gOIhw16o16i
#define FMT_WEIGHTS_BLOCKED16_IOhw16o16i IOhw16o16i
#define FMT_WEIGHTS_BLOCKED16_G_IOhw16o16i gIOhw16o16i
#elif defined(S16S16S32)
#define FMT_WEIGHTS_BLOCKED16 OIhw8o16i2o
#define FMT_WEIGHTS_BLOCKED16_G gOIhw8o16i2o
#define FMT_WEIGHTS_BLOCKED16_IOhw16o16i FMT_WEIGHTS_BLOCKED16
#define FMT_WEIGHTS_BLOCKED16_G_IOhw16o16i FMT_WEIGHTS_BLOCKED16_G
#endif
#define TEST_CASE_NAME_PREFIX BackwardData
#elif defined DIRECTION_BACKWARD_WEIGHTS
#define FMT_DATA_BLOCKED nChw8c
#define FMT_DATA_BLOCKED16 nChw16c
#define FMT_WEIGHTS_BLOCKED OIhw8i8o
#define FMT_WEIGHTS_BLOCKED_G gOIhw8i8o
#define FMT_WEIGHTS_BLOCKED16 OIhw16i16o
#define FMT_WEIGHTS_BLOCKED16_G gOIhw16i16o
#define FMT_WEIGHTS_BLOCKED16_IOhw16o16i FMT_WEIGHTS_BLOCKED16
#define FMT_WEIGHTS_BLOCKED16_G_IOhw16o16i FMT_WEIGHTS_BLOCKED16_G
#define TEST_CASE_NAME_PREFIX BackwardWeights
#endif

#define FMT_BIAS x
#define FMT_NO_BIAS format_undef

#define CONCAT_WITH_UNDERSCORE_(a,b) a ## _ ## b
#define CONCAT_WITH_UNDERSCORE(a,b) CONCAT_WITH_UNDERSCORE_(a,b)

#if defined(BIN)
#define INST_TEST_CASE_(str, ...) INSTANTIATE_TEST_CASE_P( \
        str, binary_convolution_test, ::testing::Values(__VA_ARGS__))
#define INST_TEST_CASE(str, ...) INST_TEST_CASE_( \
        CONCAT_WITH_UNDERSCORE(TEST_CASE_NAME_PREFIX, str), __VA_ARGS__)
#elif defined(DEF)
#define INST_TEST_CASE_(str, ...) INSTANTIATE_TEST_CASE_P( \
        str, deformable_convolution_test, ::testing::Values(__VA_ARGS__))
#define INST_TEST_CASE(str, ...) INST_TEST_CASE_( \
        CONCAT_WITH_UNDERSCORE(TEST_CASE_NAME_PREFIX, str), __VA_ARGS__)
#elif defined(_3D)
#define INST_TEST_CASE_3D_(str, ...) INSTANTIATE_TEST_CASE_P( \
        str, convolution_test_3d, ::testing::Values(__VA_ARGS__))
#define INST_TEST_CASE_3D(str, ...) INST_TEST_CASE_3D_( \
        CONCAT_WITH_UNDERSCORE(TEST_CASE_NAME_PREFIX, str), __VA_ARGS__)
#if defined(S8S8)
#define INST_TEST_CASE_3D_SIGNED_(str, ...) INSTANTIATE_TEST_CASE_P( \
        str, convolution_test_3d_s8, ::testing::Values(__VA_ARGS__))
#define INST_TEST_CASE_3D_SIGNED(str, ...) INST_TEST_CASE_3D_SIGNED_( \
        CONCAT_WITH_UNDERSCORE(TEST_CASE_NAME_PREFIX, str), __VA_ARGS__)
#endif
#else
#define INST_TEST_CASE_(str, ...) INSTANTIATE_TEST_CASE_P( \
        str, convolution_test, ::testing::Values(__VA_ARGS__))
#define INST_TEST_CASE(str, ...) INST_TEST_CASE_( \
        CONCAT_WITH_UNDERSCORE(TEST_CASE_NAME_PREFIX, str), __VA_ARGS__)
#if defined(S8S8)
#define INST_TEST_CASE_SIGNED_(str, ...) INSTANTIATE_TEST_CASE_P( \
        str, convolution_test_s8, ::testing::Values(__VA_ARGS__))
#define INST_TEST_CASE_SIGNED(str, ...) INST_TEST_CASE_SIGNED_( \
        CONCAT_WITH_UNDERSCORE(TEST_CASE_NAME_PREFIX, str), __VA_ARGS__)
#endif
#endif

#if defined(BIN)
#define PAD_VALUE -1.0f
#define ELTWISE_ALGORITHM mkldnn::algorithm_undef
#define DEPTHWISE_ALGORITHM mkldnn::algorithm_undef
#define BINARIZATION_ALGORITHM mkldnn::algorithm_undef
#define ELTWISE_ALPHA 0.5f
#define ELTWISE_BETA 0.1f

#if defined(WITH_SUM)
#define WITH_SUM_BOOL true
#else
#define WITH_SUM_BOOL false
#endif

#if defined(WITH_ELTWISE)
#if defined(WITH_DW_CONV)
#define PARAMS(elt_alg, src, conv1_weights, conv1_bias, conv2_weights, conv2_bias, dst, ...) \
    test_binary_convolution_dw_conv_params_t { ENGINE, ALGORITHM, elt_alg, ELTWISE_ALPHA, ELTWISE_BETA, DEPTHWISE_ALGORITHM, WITH_SUM_BOOL, BINARIZATION_ALGORITHM, \
    EXPAND_FORMATS(src, conv1_weights, conv1_bias, conv2_weights, conv2_bias, dst), \
    {__VA_ARGS__} }
#else
#define PARAMS(elt_alg, src, weights, bias, dst, ...) \
    test_binary_convolution_params_t { ENGINE, ALGORITHM, PAD_VALUE, elt_alg, ELTWISE_ALPHA, ELTWISE_BETA, DEPTHWISE_ALGORITHM, WITH_SUM_BOOL, BINARIZATION_ALGORITHM, \
    EXPAND_FORMATS(src, weights, bias, dst), \
    {__VA_ARGS__} }
#endif
#elif defined(WITH_DEPTHWISE)
#if defined(WITH_DW_CONV)
#define PARAMS(dep_alg, src, conv1_weights, conv1_bias, conv2_weights, conv2_bias, dst, ...) \
    test_binary_convolution_dw_conv_params_t { ENGINE, ALGORITHM, ELTWISE_ALGORITHM, ELTWISE_ALPHA, ELTWISE_BETA, dep_alg, WITH_SUM_BOOL, BINARIZATION_ALGORITHM, \
    EXPAND_FORMATS(src, conv1_weights, conv1_bias, conv2_weights, conv2_bias, dst), \
    {__VA_ARGS__} }
#else
#define PARAMS(dep_alg, src, weights, bias, dst, ...) \
    test_binary_convolution_params_t { ENGINE, ALGORITHM, PAD_VALUE, ELTWISE_ALGORITHM, ELTWISE_ALPHA, ELTWISE_BETA, dep_alg, WITH_SUM_BOOL, BINARIZATION_ALGORITHM, \
    EXPAND_FORMATS(src, weights, bias, dst), \
    {__VA_ARGS__} }
#endif
#elif defined(WITH_BINARIZATION)
#if defined(WITH_DW_CONV)
#define PARAMS(bin_alg, src, conv1_weights, conv1_bias, conv2_weights, conv2_bias, dst, ...) \
    test_binary_convolution_dw_conv_params_t { ENGINE, ALGORITHM, ELTWISE_ALGORITHM, ELTWISE_ALPHA, ELTWISE_BETA, DEPTHWISE_ALGORITHM, WITH_SUM_BOOL, bin_alg, \
    EXPAND_FORMATS(src, conv1_weights, conv1_bias, conv2_weights, conv2_bias, dst), \
    {__VA_ARGS__} }
#else
#define PARAMS(bin_alg, src, weights, bias, dst, ...) \
    test_binary_convolution_params_t { ENGINE, ALGORITHM, PAD_VALUE, ELTWISE_ALGORITHM, ELTWISE_ALPHA, ELTWISE_BETA, DEPTHWISE_ALGORITHM, WITH_SUM_BOOL, bin_alg, \
    EXPAND_FORMATS(src, weights, bias, dst), \
    {__VA_ARGS__} }
#endif
#else
#if defined(WITH_DW_CONV)
#define PARAMS(src, conv1_weights, conv1_bias, conv2_weights, conv2_bias, dst, ...) \
    test_binary_convolution_dw_conv_params_t { ENGINE, ALGORITHM, ELTWISE_ALGORITHM, ELTWISE_ALPHA, ELTWISE_BETA, DEPTHWISE_ALGORITHM, WITH_SUM_BOOL, BINARIZATION_ALGORITHM, \
    EXPAND_FORMATS(src, conv1_weights, conv1_bias, conv2_weights, conv2_bias, dst), \
    {__VA_ARGS__} }
#else
#define PARAMS(src, weights, bias, dst, ...) \
    test_binary_convolution_params_t { ENGINE, ALGORITHM, PAD_VALUE, ELTWISE_ALGORITHM, ELTWISE_ALPHA, ELTWISE_BETA, DEPTHWISE_ALGORITHM, WITH_SUM_BOOL, BINARIZATION_ALGORITHM, \
    EXPAND_FORMATS(src, weights, bias, dst), \
    {__VA_ARGS__} }
#endif
#endif
#elif defined(DEF)
#define PARAMS(src, offsets, weights, bias, dst, ...) \
    test_deformable_convolution_params_t { ENGINE, ALGORITHM, \
    EXPAND_FORMATS(src, offsets, weights, bias, dst), \
    {__VA_ARGS__} }
#elif defined(_3D)
#define PARAMS_3D(src, weights, bias, dst, ...) \
    test_convolution_params_t_3d { ENGINE, ALGORITHM, \
    EXPAND_FORMATS(src, weights, bias, dst), /* empty attributes */ {}, \
    {__VA_ARGS__} }
#else
#define PARAMS(src, weights, bias, dst, ...) \
    test_convolution_params_t { ENGINE, ALGORITHM, \
    EXPAND_FORMATS(src, weights, bias, dst), /* empty attributes */ {}, \
    {__VA_ARGS__} }
#endif

#define PARAMS_EXPECT_FAIL(src, weights, bias, dst, code, ...) \
    test_convolution_params_t { ENGINE, ALGORITHM, \
    EXPAND_FORMATS(src, weights, bias, dst), /* empty attributes */ {}, \
    {__VA_ARGS__}, true, code }

#define PARAMS_ATTR(src, weights, bias, dst, round_mode, scale, policy, ...) \
    test_convolution_params_t { ENGINE, ALGORITHM, \
    EXPAND_FORMATS(src, weights, bias, dst), \
    {mkldnn::round_mode, scale, test_convolution_attr_t::scale_t::policy}, \
    {__VA_ARGS__} }

#ifdef TEST_PARAM_ATTR
#include "convolution_attr.h"
#else

#if !defined(BIN) && !defined(DEF) && !defined(_3D)
#include "convolution_simple.h"
#endif

#endif
//#include "convolution_alexnet.h"
//#include "convolution_googlenet_v1.h"
//#include "convolution_googlenet_v2.h"
//#include "convolution_resnet.h"
//#include "convolution_cifar10.h"
