// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <mlas.h>

void ov_get_packedB_size(int64_t N, int64_t K) {
    MlasGemmPackBSize(N, K);
}

void ov_gemm_packB(bool transB, int64_t N, int64_t K, const uint8_t* B, size_t ldb, void* packedB) {
    MlasGemmPackB(transB ? CblasTrans : CblasNoTrans, N, K, B, ldb, packedB);
}