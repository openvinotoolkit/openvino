// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "topk_tests.hpp"

INSTANTIATE_TEST_CASE_P(
        nightly_GPU_TestsTopK, topk_test_fp32,
        ::testing::Values(
                // Params: plugin_name, in_shape, input_tensor, axis, src_k, sort, mode, out_shape, reference_val, reference_idx
                topk_test_params{ "GPU", { 3, 4 }, -1,{ 1 }, "value", "max",{ 3, 1 }, Precision::FP32},
                topk_test_params{ "GPU", { 3, 4 },  0,{ 1 }, "value", "max",{ 1, 4 }, Precision::FP32},
                topk_test_params{ "GPU", { 3, 4 }, -1,{ 1 }, "value", "min",{ 3, 1 }, Precision::FP32},
                topk_test_params{ "GPU", { 3, 4 },  0,{ 1 }, "value", "min",{ 1, 4 }, Precision::FP32},
                topk_test_params{ "GPU", { 2, 3, 128, 256 }, 1,{ 1 }, "value", "max",{ 2, 1, 128, 256 }, Precision::FP32},
                topk_test_params{ "GPU", { 3, 5, 128, 256 }, 1,{ 1 }, "index", "max",{ 3, 1, 128, 256 }, Precision::FP32},
                topk_test_params{ "GPU", { 1, 3, 129, 257 }, 1,{ 1 }, "value", "max",{ 1, 1, 129, 257 }, Precision::FP32},
                topk_test_params{ "GPU", { 2, 5, 129, 257 }, 1,{ 1 }, "index", "max",{ 2, 1, 129, 257 }, Precision::FP32},
                topk_test_params{ "GPU", { 3, 4 }, -1,{ 3 }, "value", "max",{ 3, 3 }, Precision::FP32},
                topk_test_params{ "GPU", { 3, 4 }, -1,{ 3 }, "value", "min",{ 3, 3 }, Precision::FP32},
                topk_test_params{ "GPU", { 1, 5, 1, 2 }, 1,{ 3 }, "value", "max",{ 1, 3, 1, 2 }, Precision::FP32},
                topk_test_params{ "GPU", { 1, 5, 1, 2 }, 1,{ 3 }, "value", "min",{ 1, 3, 1, 2 }, Precision::FP32},
                topk_test_params{ "GPU", { 1, 5, 1, 2 }, 1,{ 3 }, "index", "min",{ 1, 3, 1, 2 }, Precision::FP32},
                topk_test_params{ "GPU", { 1, 5, 1, 2 }, 1,{ 3 }, "index", "min",{ 1, 3, 1, 2 }, Precision::FP32},
                topk_test_params{ "GPU", { 1, 20, 12, 12 }, 1,{ 18 }, "value", "min",{ 1, 18, 12, 12 }, Precision::FP32},
                topk_test_params{ "GPU", { 1, 20, 129, 129 }, 1,{ 3 }, "value", "max",{ 1, 3, 129, 129 }, Precision::FP32},
                topk_test_params{ "GPU", { 1, 2, 2, 4 }, 3,{ 3 }, "value", "max",{ 1, 2, 2, 3 }, Precision::FP32},
                topk_test_params{ "GPU", { 1, 2, 2, 4 }, 3,{ 3 }, "index", "max",{ 1, 2, 2, 3 }, Precision::FP32},
                topk_test_params{ "GPU", { 1, 2, 2, 4 }, 3,{ 3 }, "value", "min",{ 1, 2, 2, 3 }, Precision::FP32},
                topk_test_params{ "GPU", { 1, 2, 2, 4 }, 3,{ 3 }, "index", "min",{ 1, 2, 2, 3 }, Precision::FP32},
                topk_test_params{ "GPU", { 1, 2, 2, 4 }, 3,{ 1 }, "value", "max",{ 1, 2, 2, 1 }, Precision::FP32},
                topk_test_params{ "GPU", { 1, 2, 2, 4 }, 3,{ 1 }, "index", "max",{ 1, 2, 2, 1 }, Precision::FP32},
                topk_test_params{ "GPU", { 1, 2, 4, 2 }, 2,{ 3 }, "value", "max",{ 1, 2, 3, 2 }, Precision::FP32},
                topk_test_params{ "GPU", { 1, 2, 4, 2 }, 2,{ 3 }, "index", "max",{ 1, 2, 3, 2 }, Precision::FP32},
                topk_test_params{ "GPU", { 1, 2, 4, 2 }, 2,{ 3 }, "value", "min",{ 1, 2, 3, 2 }, Precision::FP32},
                topk_test_params{ "GPU", { 1, 2, 4, 2 }, 2,{ 3 }, "index", "min",{ 1, 2, 3, 2 }, Precision::FP32},
                topk_test_params{ "GPU", { 1, 2, 2, 4 }, 3,{ 3 }, "index", "min",{ 1, 2, 2, 3 }, Precision::FP32},
                topk_test_params{ "GPU", { 1, 2, 2, 4 }, 3,{ 3 }, "index", "max",{ 1, 2, 2, 3 }, Precision::FP32},
                topk_test_params{ "GPU", { 1, 2, 2, 4 }, 3,{ 3 }, "value", "min",{ 1, 2, 2, 3 }, Precision::FP32},
                topk_test_params{ "GPU", { 1, 2, 2, 4 }, 3,{ 3 }, "value", "max",{ 1, 2, 2, 3 }, Precision::FP32},
                topk_test_params{ "GPU", { 1, 20, 32, 32 }, 1,{ 18 }, "index", "max",{ 1, 18, 32, 32 }, Precision::FP32},
                topk_test_params{ "GPU", { 1, 20, 129, 129 }, 1,{ 18 }, "index", "max",{ 1, 18, 129, 129 }, Precision::FP32},
                topk_test_params{ "GPU", { 1, 20, 32, 32 }, 1,{ 18 }, "index", "min",{ 1, 18, 32, 32 }, Precision::FP32},
                topk_test_params{ "GPU", { 1, 20, 129, 129 }, 1,{ 18 }, "index", "min",{ 1, 18, 129, 129 }, Precision::FP32}
        ));

INSTANTIATE_TEST_CASE_P(
        nightly_GPU_TestsTopK, topk_test_int32,
        ::testing::Values(
                // Params: plugin_name, in_shape, input_tensor, axis, src_k, sort, mode, out_shape, reference_val, reference_idx
                topk_test_params{ "GPU", { 3, 4 }, -1,{ 1 }, "value", "max",{ 3, 1 }, Precision::I32},
                topk_test_params{ "GPU", { 3, 4 },  0,{ 1 }, "value", "max",{ 1, 4 }, Precision::I32},
                topk_test_params{ "GPU", { 3, 4 }, -1,{ 1 }, "value", "min",{ 3, 1 }, Precision::I32},
                topk_test_params{ "GPU", { 3, 4 },  0,{ 1 }, "value", "min",{ 1, 4 }, Precision::I32},
                topk_test_params{ "GPU", { 2, 3, 128, 256 }, 1,{ 1 }, "value", "max",{ 2, 1, 128, 256 }, Precision::I32},
                topk_test_params{ "GPU", { 3, 5, 128, 256 }, 1,{ 1 }, "index", "max",{ 3, 1, 128, 256 }, Precision::I32},
                topk_test_params{ "GPU", { 1, 3, 129, 257 }, 1,{ 1 }, "value", "max",{ 1, 1, 129, 257 }, Precision::I32},
                topk_test_params{ "GPU", { 2, 5, 129, 257 }, 1,{ 1 }, "index", "max",{ 2, 1, 129, 257 }, Precision::I32},
                topk_test_params{ "GPU", { 3, 4 }, -1,{ 3 }, "value", "max",{ 3, 3 }, Precision::I32},
                topk_test_params{ "GPU", { 3, 4 }, -1,{ 3 }, "value", "min",{ 3, 3 }, Precision::I32},
                topk_test_params{ "GPU", { 1, 5, 1, 2 }, 1,{ 3 }, "value", "max",{ 1, 3, 1, 2 }, Precision::I32},
                topk_test_params{ "GPU", { 1, 5, 1, 2 }, 1,{ 3 }, "value", "min",{ 1, 3, 1, 2 }, Precision::I32},
                topk_test_params{ "GPU", { 1, 5, 1, 2 }, 1,{ 3 }, "index", "min",{ 1, 3, 1, 2 }, Precision::I32},
                topk_test_params{ "GPU", { 1, 5, 1, 2 }, 1,{ 3 }, "index", "min",{ 1, 3, 1, 2 }, Precision::I32},
                topk_test_params{ "GPU", { 1, 20, 12, 12 }, 1,{ 18 }, "value", "min",{ 1, 18, 12, 12 }, Precision::I32},
                topk_test_params{ "GPU", { 1, 20, 129, 129 }, 1,{ 3 }, "value", "max",{ 1, 3, 129, 129 }, Precision::I32},
                topk_test_params{ "GPU", { 1, 2, 2, 4 }, 3,{ 3 }, "value", "max",{ 1, 2, 2, 3 }, Precision::I32},
                topk_test_params{ "GPU", { 1, 2, 2, 4 }, 3,{ 3 }, "index", "max",{ 1, 2, 2, 3 }, Precision::I32},
                topk_test_params{ "GPU", { 1, 2, 2, 4 }, 3,{ 3 }, "value", "min",{ 1, 2, 2, 3 }, Precision::I32},
                topk_test_params{ "GPU", { 1, 2, 2, 4 }, 3,{ 3 }, "index", "min",{ 1, 2, 2, 3 }, Precision::I32},
                topk_test_params{ "GPU", { 1, 2, 2, 4 }, 3,{ 1 }, "value", "max",{ 1, 2, 2, 1 }, Precision::I32},
                topk_test_params{ "GPU", { 1, 2, 2, 4 }, 3,{ 1 }, "index", "max",{ 1, 2, 2, 1 }, Precision::I32},
                topk_test_params{ "GPU", { 1, 2, 4, 2 }, 2,{ 3 }, "value", "max",{ 1, 2, 3, 2 }, Precision::I32},
                topk_test_params{ "GPU", { 1, 2, 4, 2 }, 2,{ 3 }, "index", "max",{ 1, 2, 3, 2 }, Precision::I32},
                topk_test_params{ "GPU", { 1, 2, 4, 2 }, 2,{ 3 }, "value", "min",{ 1, 2, 3, 2 }, Precision::I32},
                topk_test_params{ "GPU", { 1, 2, 4, 2 }, 2,{ 3 }, "index", "min",{ 1, 2, 3, 2 }, Precision::I32},
                topk_test_params{ "GPU", { 1, 2, 2, 4 }, 3,{ 3 }, "index", "min",{ 1, 2, 2, 3 }, Precision::I32},
                topk_test_params{ "GPU", { 1, 2, 2, 4 }, 3,{ 3 }, "index", "max",{ 1, 2, 2, 3 }, Precision::I32},
                topk_test_params{ "GPU", { 1, 2, 2, 4 }, 3,{ 3 }, "value", "min",{ 1, 2, 2, 3 }, Precision::I32},
                topk_test_params{ "GPU", { 1, 2, 2, 4 }, 3,{ 3 }, "value", "max",{ 1, 2, 2, 3 }, Precision::I32},
                topk_test_params{ "GPU", { 1, 20, 32, 32 }, 1,{ 18 }, "index", "max",{ 1, 18, 32, 32 }, Precision::I32},
                topk_test_params{ "GPU", { 1, 20, 129, 129 }, 1,{ 18 }, "index", "max",{ 1, 18, 129, 129 }, Precision::I32},
                topk_test_params{ "GPU", { 1, 20, 32, 32 }, 1,{ 18 }, "index", "min",{ 1, 18, 32, 32 }, Precision::I32},
                topk_test_params{ "GPU", { 1, 20, 129, 129 }, 1,{ 18 }, "index", "min",{ 1, 18, 129, 129 }, Precision::I32}
        ));
