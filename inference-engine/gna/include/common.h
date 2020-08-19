/*
 INTEL CONFIDENTIAL
 Copyright 2019 Intel Corporation.

 The source code contained or described herein and all documents related
 to the source code ("Material") are owned by Intel Corporation or its suppliers
 or licensors. Title to the Material remains with Intel Corporation or its suppliers
 and licensors. The Material may contain trade secrets and proprietary
 and confidential information of Intel Corporation and its suppliers and licensors,
 and is protected by worldwide copyright and trade secret laws and treaty provisions.
 No part of the Material may be used, copied, reproduced, modified, published,
 uploaded, posted, transmitted, distributed, or disclosed in any way without Intel's
 prior express written permission.

 No license under any patent, copyright, trade secret or other intellectual
 property right is granted to or conferred upon you by disclosure or delivery
 of the Materials, either expressly, by implication, inducement, estoppel
 or otherwise. Any license under such intellectual property rights must
 be express and approved by Intel in writing.

 Unless otherwise agreed by Intel in writing, you may not remove or alter this notice
 or any other notice embedded in Materials by Intel or Intel's suppliers or licensors
 in any way.
*/

#pragma once

#include "gna-api-types-xnn.h"
#include "gna-api-types-gmm.h"
#include "gna-api.h"

#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <functional>

#if defined(__GNUC__) && !defined(__INTEL_COMPILER)
#include <mm_malloc.h>
#endif

/**
 * Data alignment for intrinsics
 */
#define INTRIN_ALIGN 0x40

/**
 * GNA main memory required alignment size
 */
#ifndef PAGE_SIZE // on Android OS PAGE_SIZE is defined in <sys/user.h>
const uint32_t PAGE_SIZE = 0x1000;
#endif //PAGE_SIZE

const uint32_t ui32_1 = 1;
const uint32_t ui32_0 = 0;
const uint32_t ui32_UINT8_MAX = UINT8_MAX;

#define _gna_malloc(a)    _mm_malloc(a, PAGE_SIZE)
#define _kernel_malloc(a) _mm_malloc(a, INTRIN_ALIGN)
#define _gna_free(a)      _mm_free(a)

/**
 * shorter aliases for official GMM API types
 */
typedef gna_layer_operation         nn_operation;
typedef intel_compound_bias_t       nn_bias_c;
typedef intel_weight_scaling_factor_t nn_scaling;
typedef intel_bias_t                nn_bias_s;
typedef intel_affine_func_t         nn_func_affine;
typedef intel_affine_multibias_func_t nn_func_affine_multi;
typedef intel_pwl_segment_t         nn_pwl_seg;
typedef intel_pwl_func_t            nn_func_pwl;
typedef intel_nnet_layer_t          nn_layer;
typedef intel_affine_layer_t        nn_layer_affine;
typedef intel_affine_multibias_layer_t nn_layer_affine_multi;
typedef intel_recurrent_layer_t     nn_layer_recurrent;
typedef gna_convolutional_fused_layer_2d  nn_layer_cnn2d;
typedef gna_pooling_layer_2d        nn_layer_pool2d;
typedef intel_copy_layer_t          nn_layer_copy;
typedef intel_pool_type_t           nn_pool_type;
typedef intel_convolutional_layer_t nn_layer_conv;
typedef intel_gna_status_t          status_t;

typedef enum _TransformOperation
{
    ActivationTransform,
    AffineTransform,
    AffineDiagonalTransform,
    AffineMultibiasTransform,
    ConvolutionalTranform1D,
    ConvolutionalTransform2D,
    CopyTransform,
    TransposeTransform,
    GmmTransform,
    PoolingTranform1D,
    PoolingTransform2D,
    RecurrentTransform,
    TransformOperationCount,
} TransformOperation;

template<typename T, typename X>
T RoundUp(T number, X significanceIn)
{
    auto const significance = static_cast<T>(significanceIn);
    if (0 == significance)
    {
        return number;
    }
    else
    {
        return ((number + significance - 1) / significance) * significance;
    }
}

template<typename T>
T GnaFloor(T number, T significance)
{
    return static_cast<T>((number / significance) * significance);
}

template<typename T>
T GnaCeilDiv(T number, T divider)
{
    return static_cast<T>(((number) + divider - 1) / divider);
}

inline int32_t getBias(const void* ptr, uint32_t bytesPerElement, uint32_t idx = 0)
{
    switch (bytesPerElement)
    {
    case 1:
        return ((int8_t*)ptr)[idx];
    case 2:
        return ((int16_t*)ptr)[idx];
    case 4:
        return ((int32_t*)ptr)[idx];
    default:
        return 0;
    }
}
