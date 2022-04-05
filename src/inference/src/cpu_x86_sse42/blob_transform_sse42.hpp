// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stdint.h>
#include <stdlib.h>

namespace InferenceEngine {

//------------------------------------------------------------------------
//
// Blob-copy primitives namually vectored for SSE 4.2 (w/o OpenMP threads)
//
//------------------------------------------------------------------------

void blob_copy_4d_split_u8c3(const uint8_t* src_ptr,
                             uint8_t* dst_ptr,
                             size_t N_src_stride,
                             size_t H_src_stride,
                             size_t N_dst_stride,
                             size_t H_dst_stride,
                             size_t C_dst_stride,
                             int N,
                             int H,
                             int W);

void blob_copy_4d_split_f32c3(const float* src_ptr,
                              float* dst_ptr,
                              size_t N_src_stride,
                              size_t H_src_stride,
                              size_t N_dst_stride,
                              size_t H_dst_stride,
                              size_t C_dst_stride,
                              int N,
                              int H,
                              int W);

void blob_copy_4d_merge_u8c3(const uint8_t* src_ptr,
                             uint8_t* dst_ptr,
                             size_t N_src_stride,
                             size_t H_src_stride,
                             size_t C_src_stride,
                             size_t N_dst_stride,
                             size_t H_dst_stride,
                             int N,
                             int H,
                             int W);

void blob_copy_4d_merge_f32c3(const float* src_ptr,
                              float* dst_ptr,
                              size_t N_src_stride,
                              size_t H_src_stride,
                              size_t C_src_stride,
                              size_t N_dst_stride,
                              size_t H_dst_stride,
                              int N,
                              int H,
                              int W);

void blob_copy_5d_split_u8c3(const uint8_t* src_ptr,
                             uint8_t* dst_ptr,
                             size_t N_src_stride,
                             size_t D_src_stride,
                             size_t H_src_stride,
                             size_t N_dst_stride,
                             size_t D_dst_stride,
                             size_t H_dst_stride,
                             size_t C_dst_stride,
                             int N,
                             int D,
                             int H,
                             int W);

void blob_copy_5d_split_f32c3(const float* src_ptr,
                              float* dst_ptr,
                              size_t N_src_stride,
                              size_t D_src_stride,
                              size_t H_src_stride,
                              size_t N_dst_stride,
                              size_t D_dst_stride,
                              size_t H_dst_stride,
                              size_t C_dst_stride,
                              int N,
                              int D,
                              int H,
                              int W);

void blob_copy_5d_merge_u8c3(const uint8_t* src_ptr,
                             uint8_t* dst_ptr,
                             size_t N_src_stride,
                             size_t D_src_stride,
                             size_t H_src_stride,
                             size_t C_src_stride,
                             size_t N_dst_stride,
                             size_t D_dst_stride,
                             size_t H_dst_stride,
                             int N,
                             int D,
                             int H,
                             int W);

void blob_copy_5d_merge_f32c3(const float* src_ptr,
                              float* dst_ptr,
                              size_t N_src_stride,
                              size_t D_src_stride,
                              size_t H_src_stride,
                              size_t C_src_stride,
                              size_t N_dst_stride,
                              size_t D_dst_stride,
                              size_t H_dst_stride,
                              int N,
                              int D,
                              int H,
                              int W);

}  // namespace InferenceEngine
