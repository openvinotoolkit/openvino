// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/infer_request/set_io_blob_precision.hpp"

using namespace BehaviorTestsDefinitions;
using namespace InferenceEngine;

const std::vector<Precision> precisionSet = {
    Precision::U8,   Precision::I8,
    Precision::U16,  Precision::I16,
    Precision::U32,  Precision::I32,
    Precision::U64,  Precision::I64,
    Precision::BF16, Precision::FP16,
    Precision::FP32, Precision::FP64,
    Precision::BOOL
};

const std::vector<setType> typeSet = {setType::INPUT, setType::OUTPUT, setType::BOTH};

const auto params = ::testing::Combine(::testing::ValuesIn(precisionSet),
                                       ::testing::ValuesIn(precisionSet),
                                       ::testing::ValuesIn(typeSet),
                                       ::testing::Values(ov::test::utils::DEVICE_CPU));

INSTANTIATE_TEST_SUITE_P(smoke_SetBlobCPU, SetBlobTest, params, SetBlobTest::getTestCaseName);
