/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#ifndef COMMON_IMPL_REGISTRATION_HPP
#define COMMON_IMPL_REGISTRATION_HPP

#include "oneapi/dnnl/dnnl_config.h"

// Primitives section

// Note:
// `_P` is a mandatory suffix for macros. This is to avoid a conflict with
// `REG_BINARY`, Windows-defined macro.

#if BUILD_PRIMITIVE_ALL || (BUILD_BATCH_NORMALIZATION)
#define REG_BNORM_P_FWD(...) __VA_ARGS__
#else
#define REG_BNORM_P_FWD(...)
#endif

#if BUILD_PRIMITIVE_ALL || (BUILD_BATCH_NORMALIZATION && BUILD_TRAINING)
#define REG_BNORM_P_BWD(...) __VA_ARGS__
#else
#define REG_BNORM_P_BWD(...)
#endif

#if BUILD_PRIMITIVE_ALL || (BUILD_BINARY)
#define REG_BINARY_P(...) __VA_ARGS__
#else
#define REG_BINARY_P(...)
#endif

#if BUILD_PRIMITIVE_ALL || (BUILD_CONCAT)
#define REG_CONCAT_P(...) __VA_ARGS__
#else
#define REG_CONCAT_P(...)
#endif

#if BUILD_PRIMITIVE_ALL || (BUILD_CONVOLUTION)
#define REG_CONV_P_FWD(...) __VA_ARGS__
#else
#define REG_CONV_P_FWD(...)
#endif

#if BUILD_PRIMITIVE_ALL || (BUILD_CONVOLUTION && BUILD_TRAINING)
#define REG_CONV_P_BWD_D(...) __VA_ARGS__
#define REG_CONV_P_BWD_W(...) __VA_ARGS__
#else
#define REG_CONV_P_BWD_D(...)
#define REG_CONV_P_BWD_W(...)
#endif

#if BUILD_PRIMITIVE_ALL || (BUILD_DECONVOLUTION)
#define REG_DECONV_P_FWD(...) __VA_ARGS__
// This case is special, it requires handling of convolution_bwd_d internally
// since major optimizations are based on convolution implementations.
#ifndef REG_CONV_P_BWD_D
#error "REG_CONV_P_BWD_D is not defined. Check that convolution " \
       "is defined prior deconvolution."
#endif
#undef REG_CONV_P_BWD_D
#define REG_CONV_P_BWD_D(...) __VA_ARGS__
#else
#define REG_DECONV_P_FWD(...)
#endif

#if BUILD_PRIMITIVE_ALL || (BUILD_DECONVOLUTION && BUILD_TRAINING)
#define REG_DECONV_P_BWD(...) __VA_ARGS__
// This case is special, it requires handling of convolution_fwd and
// convolution_bwd_w internally since major optimizations are based on
// convolution implementations.
#ifndef REG_CONV_P_FWD
#error "REG_CONV_P_FWD is not defined. Check that convolution " \
       "is defined prior deconvolution."
#endif
#undef REG_CONV_P_FWD
#define REG_CONV_P_FWD(...) __VA_ARGS__

#ifndef REG_CONV_P_BWD_W
#error "REG_CONV_P_BWD_W is not defined. Check that convolution " \
       "is defined prior deconvolution."
#endif
#undef REG_CONV_P_BWD_W
#define REG_CONV_P_BWD_W(...) __VA_ARGS__
#else
#define REG_DECONV_P_BWD(...)
#endif

#if BUILD_PRIMITIVE_ALL || (BUILD_ELTWISE)
#define REG_ELTWISE_P_FWD(...) __VA_ARGS__
#else
#define REG_ELTWISE_P_FWD(...)
#endif

#if BUILD_PRIMITIVE_ALL || (BUILD_ELTWISE && BUILD_TRAINING)
#define REG_ELTWISE_P_BWD(...) __VA_ARGS__
#else
#define REG_ELTWISE_P_BWD(...)
#endif

#if BUILD_PRIMITIVE_ALL || (BUILD_INNER_PRODUCT)
#define REG_IP_P_FWD(...) __VA_ARGS__
#else
#define REG_IP_P_FWD(...)
#endif

#if BUILD_PRIMITIVE_ALL || (BUILD_INNER_PRODUCT && BUILD_TRAINING)
#define REG_IP_P_BWD(...) __VA_ARGS__
#else
#define REG_IP_P_BWD(...)
#endif

#if BUILD_PRIMITIVE_ALL || (BUILD_LAYER_NORMALIZATION)
#define REG_LNORM_P_FWD(...) __VA_ARGS__
#else
#define REG_LNORM_P_FWD(...)
#endif

#if BUILD_PRIMITIVE_ALL || (BUILD_LAYER_NORMALIZATION && BUILD_TRAINING)
#define REG_LNORM_P_BWD(...) __VA_ARGS__
#else
#define REG_LNORM_P_BWD(...)
#endif

#if BUILD_PRIMITIVE_ALL || (BUILD_LRN)
#define REG_LRN_P_FWD(...) __VA_ARGS__
#else
#define REG_LRN_P_FWD(...)
#endif

#if BUILD_PRIMITIVE_ALL || (BUILD_LRN && BUILD_TRAINING)
#define REG_LRN_P_BWD(...) __VA_ARGS__
#else
#define REG_LRN_P_BWD(...)
#endif

#if BUILD_PRIMITIVE_ALL || (BUILD_MATMUL)
#define REG_MATMUL_P(...) __VA_ARGS__
#else
#define REG_MATMUL_P(...)
#endif

#if BUILD_PRIMITIVE_ALL || (BUILD_POOLING)
#define REG_POOLING_P_FWD(...) __VA_ARGS__
#else
#define REG_POOLING_P_FWD(...)
#endif

#if BUILD_PRIMITIVE_ALL || (BUILD_POOLING && BUILD_TRAINING)
#define REG_POOLING_P_BWD(...) __VA_ARGS__
#else
#define REG_POOLING_P_BWD(...)
#endif

#if BUILD_PRIMITIVE_ALL || (BUILD_PRELU)
#define REG_PRELU_P_FWD(...) __VA_ARGS__
#else
#define REG_PRELU_P_FWD(...)
#endif

#if BUILD_PRIMITIVE_ALL || (BUILD_PRELU && BUILD_TRAINING)
#define REG_PRELU_P_BWD(...) __VA_ARGS__
#else
#define REG_PRELU_P_BWD(...)
#endif

#if BUILD_PRIMITIVE_ALL || (BUILD_REDUCTION)
#define REG_REDUCTION_P(...) __VA_ARGS__
#else
#define REG_REDUCTION_P(...)
#endif

#if BUILD_PRIMITIVE_ALL || (BUILD_REORDER)
#define REG_REORDER_P(...) __VA_ARGS__
#else
#define REG_REORDER_P(...)
#endif

#if BUILD_PRIMITIVE_ALL || (BUILD_RESAMPLING)
#define REG_RESAMPLING_P_FWD(...) __VA_ARGS__
#else
#define REG_RESAMPLING_P_FWD(...)
#endif

#if BUILD_PRIMITIVE_ALL || (BUILD_RESAMPLING && BUILD_TRAINING)
#define REG_RESAMPLING_P_BWD(...) __VA_ARGS__
#else
#define REG_RESAMPLING_P_BWD(...)
#endif

#if BUILD_PRIMITIVE_ALL || (BUILD_RNN)
#define REG_RNN_P_FWD(...) __VA_ARGS__
#else
#define REG_RNN_P_FWD(...)
#endif

#if BUILD_PRIMITIVE_ALL || (BUILD_RNN && BUILD_TRAINING)
#define REG_RNN_P_BWD(...) __VA_ARGS__
#else
#define REG_RNN_P_BWD(...)
#endif

#if BUILD_PRIMITIVE_ALL || (BUILD_SHUFFLE)
#define REG_SHUFFLE_P(...) __VA_ARGS__
#else
#define REG_SHUFFLE_P(...)
#endif

#if BUILD_PRIMITIVE_ALL || (BUILD_SOFTMAX)
#define REG_SOFTMAX_P_FWD(...) __VA_ARGS__
#else
#define REG_SOFTMAX_P_FWD(...)
#endif

#if BUILD_PRIMITIVE_ALL || (BUILD_SOFTMAX && BUILD_TRAINING)
#define REG_SOFTMAX_P_BWD(...) __VA_ARGS__
#else
#define REG_SOFTMAX_P_BWD(...)
#endif

#if BUILD_PRIMITIVE_ALL || (BUILD_SUM)
#define REG_SUM_P(...) __VA_ARGS__
#else
#define REG_SUM_P(...)
#endif

#endif
