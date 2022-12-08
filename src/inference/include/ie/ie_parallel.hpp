// Copyright (C) 2018-2022 Intel Corporation
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

#include "openvino/core/parallel.hpp"

#define IE_THREAD_TBB      0
#define IE_THREAD_OMP      1
#define IE_THREAD_SEQ      2
#define IE_THREAD_TBB_AUTO 3

namespace InferenceEngine {

using ov::parallel_it_init;
using ov::parallel_it_step;
using ov::parallel_nt;
using ov::parallel_nt_static;
using ov::parallel_sort;
using ov::parallel_sum;
using ov::parallel_sum2d;
using ov::parallel_sum3d;
using ov::splitter;

namespace details {

using ov::details::call_with_args;
using ov::details::num_of_lambda_args;

}  // namespace details

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

}  // namespace InferenceEngine
