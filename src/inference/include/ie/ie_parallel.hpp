// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Contains declarations and definitions for sequential and multi-threading implementations.
 *
 * Multi-threading support is implemented in two variants: using the Threading Building Blocks library and OpenMP*
 * product. To build a particular implementation, use the corresponding identifier: IE_THREAD_TBB, IE_THREAD_TBB_AUTO,
 * IE_THREAD_OMP or IE_THREAD_SEQ.
 *
 * @file ie_parallel.hpp
 */

#pragma once

#if !defined(IN_OV_COMPONENT) && !defined(IE_LEGACY_HEADER_INCLUDED)
#    define IE_LEGACY_HEADER_INCLUDED
#    ifdef _MSC_VER
#        pragma message( \
            "The Inference Engine API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    else
#        warning("The Inference Engine API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    endif
#endif

#include "openvino/core/parallel.hpp"

#define IE_THREAD_TBB      OV_THREAD_TBB
#define IE_THREAD_OMP      OV_THREAD_OMP
#define IE_THREAD_SEQ      OV_THREAD_SEQ
#define IE_THREAD_TBB_AUTO OV_THREAD_TBB_AUTO

namespace InferenceEngine {

using ov::for_1d;
using ov::for_2d;
using ov::for_3d;
using ov::for_4d;
using ov::for_5d;
using ov::for_6d;
using ov::parallel_for;
using ov::parallel_for2d;
using ov::parallel_for3d;
using ov::parallel_for4d;
using ov::parallel_for5d;
using ov::parallel_for6d;
using ov::parallel_it_init;
using ov::parallel_it_step;
using ov::parallel_nt;
using ov::parallel_nt_static;
using ov::parallel_sort;
using ov::parallel_sum;
using ov::parallel_sum2d;
using ov::parallel_sum3d;
using ov::splitter;

}  // namespace InferenceEngine
