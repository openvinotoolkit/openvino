// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "resample_tests.hpp"

INSTANTIATE_TEST_CASE_P(
        smoke_CPU_TestsResample, ResampleTests,
        ::testing::Values(
                // 4D nearest
                resample_test_params{"CPU", {2, 64, 15, 25}, 1.f,   "caffe.ResampleParameter.NEAREST"},
                resample_test_params{"CPU", {2, 64, 10, 20}, 0.25f, "caffe.ResampleParameter.NEAREST"},
                resample_test_params{"CPU", {1, 1, 10, 20},  0.5f,  "caffe.ResampleParameter.NEAREST"},
                resample_test_params{"CPU", {2, 3, 15, 25},  1.f,   "caffe.ResampleParameter.NEAREST"},
                resample_test_params{"CPU", {2, 3, 10, 20},  0.25f, "caffe.ResampleParameter.NEAREST"},
                resample_test_params{"CPU", {1, 1, 10, 13},  0.52f, "caffe.ResampleParameter.NEAREST"},
                //// 4D linear
                resample_test_params{"CPU", {2, 64, 15, 25}, 1.f,   "caffe.ResampleParameter.LINEAR"},
                resample_test_params{"CPU", {2, 64, 10, 20}, 0.25f, "caffe.ResampleParameter.LINEAR"},
                resample_test_params{"CPU", {1, 1, 15, 25},  0.5,   "caffe.ResampleParameter.LINEAR"},
                resample_test_params{"CPU", {1, 3, 15, 25},  0.5,   "caffe.ResampleParameter.LINEAR"},
                resample_test_params{"CPU", {2, 5, 3, 3},    3.0f,  "caffe.ResampleParameter.LINEAR"},
                resample_test_params{"CPU", {2, 4, 10, 20},  2.0f,  "caffe.ResampleParameter.LINEAR"},
                resample_test_params{"CPU", {2, 20, 30, 30}, 3.0f,  "caffe.ResampleParameter.LINEAR"},
                resample_test_params{"CPU", {2, 20, 3, 6},   3.0f,  "caffe.ResampleParameter.LINEAR"},
                //// 5D nearest
                resample_test_params{ "CPU", {1, 64, 20, 15, 25}, 1.f,   "caffe.ResampleParameter.NEAREST" },
                resample_test_params{ "CPU", {1, 64, 15, 10, 20}, 0.25f, "caffe.ResampleParameter.NEAREST" },
                resample_test_params{ "CPU", {1, 64, 10, 10, 20}, 0.5f,  "caffe.ResampleParameter.NEAREST" },
                resample_test_params{ "CPU", {1, 3, 20, 15, 25},  1.f,   "caffe.ResampleParameter.NEAREST" },
                resample_test_params{ "CPU", {1, 3, 15, 10, 20},  0.25f, "caffe.ResampleParameter.NEAREST" },
                resample_test_params{ "CPU", {2, 64, 20, 15, 25}, 1.f,   "caffe.ResampleParameter.NEAREST" },
                resample_test_params{ "CPU", {2, 64, 15, 10, 20}, 0.25f, "caffe.ResampleParameter.NEAREST" },
                resample_test_params{ "CPU", {2, 64, 10, 10, 20}, 0.5f,  "caffe.ResampleParameter.NEAREST" },
                resample_test_params{ "CPU", {2, 3, 20, 15, 25},  1.f,   "caffe.ResampleParameter.NEAREST" },
                resample_test_params{ "CPU", {2, 3, 15, 10, 20},  0.25f, "caffe.ResampleParameter.NEAREST" },
                // 5D linear
                resample_test_params{ "CPU", {1, 8, 5, 2, 4},     0.2f,  "caffe.ResampleParameter.LINEAR" },
                resample_test_params{ "CPU", {1, 8, 10, 10, 20},  0.25f, "caffe.ResampleParameter.LINEAR" },
                resample_test_params{ "CPU", {1, 2, 16, 12, 20},  4.f,   "caffe.ResampleParameter.LINEAR" },
                resample_test_params{ "CPU", {2, 16, 15, 10, 20}, 1.f,   "caffe.ResampleParameter.LINEAR" },
                resample_test_params{ "CPU", {2, 2, 4, 10, 20},   0.25f, "caffe.ResampleParameter.LINEAR" },
                resample_test_params{ "CPU", {2, 4, 15, 10, 20},  1.f,   "caffe.ResampleParameter.LINEAR" },
                resample_test_params{ "CPU", {2, 8, 16, 12, 20},  4.f,   "caffe.ResampleParameter.LINEAR" },
                resample_test_params{ "CPU", {2, 16, 10, 10, 20}, 0.25f, "caffe.ResampleParameter.LINEAR" }));