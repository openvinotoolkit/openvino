// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/plugin/set_preprocess.hpp"
#include "api_conformance_helpers.hpp"

namespace {

using namespace BehaviorTestsDefinitions;
using namespace ov::test::conformance;

const std::vector<InferenceEngine::Precision> netPrecisionsPreprocess = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
};

const std::vector<std::map<std::string, std::string>> configs = {
        {},
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, InferRequestPreprocessTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(netPrecisionsPreprocess),
                                ::testing::Values(ov::test::conformance::targetDevice),
                                ::testing::ValuesIn(configs)),
                         InferRequestPreprocessTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests, InferRequestPreprocessTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(netPrecisionsPreprocess),
                                ::testing::Values(CommonTestUtils::DEVICE_HETERO),
                                ::testing::ValuesIn(ov::test::conformance::generate_configs(CommonTestUtils::DEVICE_HETERO))),
                         InferRequestPreprocessTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, InferRequestPreprocessTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(netPrecisionsPreprocess),
                                ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                ::testing::ValuesIn(ov::test::conformance::generate_configs(CommonTestUtils::DEVICE_MULTI))),
                         InferRequestPreprocessTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, InferRequestPreprocessTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(netPrecisionsPreprocess),
                                ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                ::testing::ValuesIn(ov::test::conformance::generate_configs(CommonTestUtils::DEVICE_AUTO))),
                         InferRequestPreprocessTest::getTestCaseName);


const std::vector<InferenceEngine::Precision> ioPrecisionsPreprocess = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::U8
};
const std::vector<InferenceEngine::Layout> netLayoutsPreprocess = {
    InferenceEngine::Layout::NCHW,
    // InferenceEngine::Layout::NHWC
};

const std::vector<InferenceEngine::Layout> ioLayoutsPreprocess = {
    InferenceEngine::Layout::NCHW,
    InferenceEngine::Layout::NHWC
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, InferRequestPreprocessConversionTest,
                    ::testing::Combine(
                            ::testing::ValuesIn(netPrecisionsPreprocess),
                            ::testing::ValuesIn(ioPrecisionsPreprocess),
                            ::testing::ValuesIn(ioPrecisionsPreprocess),
                            ::testing::ValuesIn(netLayoutsPreprocess),
                            ::testing::ValuesIn(ioLayoutsPreprocess),
                            ::testing::ValuesIn(ioLayoutsPreprocess),
                            ::testing::Bool(),
                            ::testing::Bool(),
                            ::testing::Values(ov::test::conformance::targetDevice),
                            ::testing::ValuesIn(configs)),
                    InferRequestPreprocessConversionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, InferRequestPreprocessDynamicallyInSetBlobTest,
                    ::testing::Combine(
                            ::testing::ValuesIn(netPrecisionsPreprocess),
                            ::testing::Bool(),
                            ::testing::Bool(),
                            ::testing::ValuesIn(netLayoutsPreprocess),
                            ::testing::Bool(),
                            ::testing::Bool(),
                            ::testing::Values(true), // only SetBlob
                            ::testing::Values(true), // only SetBlob
                            ::testing::Values(ov::test::conformance::targetDevice),
                            ::testing::ValuesIn(configs)),
                    InferRequestPreprocessDynamicallyInSetBlobTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests, InferRequestPreprocessConversionTest,
                    ::testing::Combine(
                            ::testing::ValuesIn(netPrecisionsPreprocess),
                            ::testing::ValuesIn(ioPrecisionsPreprocess),
                            ::testing::ValuesIn(ioPrecisionsPreprocess),
                            ::testing::ValuesIn(netLayoutsPreprocess),
                            ::testing::ValuesIn(ioLayoutsPreprocess),
                            ::testing::ValuesIn(ioLayoutsPreprocess),
                            ::testing::Bool(),
                            ::testing::Bool(),
                            ::testing::Values(CommonTestUtils::DEVICE_HETERO),
                            ::testing::ValuesIn(ov::test::conformance::generate_configs(CommonTestUtils::DEVICE_HETERO))),
                    InferRequestPreprocessConversionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests, InferRequestPreprocessDynamicallyInSetBlobTest,
                    ::testing::Combine(
                            ::testing::ValuesIn(netPrecisionsPreprocess),
                            ::testing::Bool(),
                            ::testing::Bool(),
                            ::testing::ValuesIn(netLayoutsPreprocess),
                            ::testing::Bool(),
                            ::testing::Bool(),
                            ::testing::Values(true), // only SetBlob
                            ::testing::Values(true), // only SetBlob
                            ::testing::Values(CommonTestUtils::DEVICE_HETERO),
                            ::testing::ValuesIn(ov::test::conformance::generate_configs(CommonTestUtils::DEVICE_HETERO))),
                    InferRequestPreprocessDynamicallyInSetBlobTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, InferRequestPreprocessConversionTest,
                    ::testing::Combine(
                            ::testing::ValuesIn(netPrecisionsPreprocess),
                            ::testing::ValuesIn(ioPrecisionsPreprocess),
                            ::testing::ValuesIn(ioPrecisionsPreprocess),
                            ::testing::ValuesIn(netLayoutsPreprocess),
                            ::testing::ValuesIn(ioLayoutsPreprocess),
                            ::testing::ValuesIn(ioLayoutsPreprocess),
                            ::testing::Bool(),
                            ::testing::Bool(),
                            ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                            ::testing::ValuesIn(ov::test::conformance::generate_configs(CommonTestUtils::DEVICE_MULTI))),
                    InferRequestPreprocessConversionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, InferRequestPreprocessDynamicallyInSetBlobTest,
                    ::testing::Combine(
                            ::testing::ValuesIn(netPrecisionsPreprocess),
                            ::testing::Bool(),
                            ::testing::Bool(),
                            ::testing::ValuesIn(netLayoutsPreprocess),
                            ::testing::Bool(),
                            ::testing::Bool(),
                            ::testing::Values(true), // only SetBlob
                            ::testing::Values(true), // only SetBlob
                            ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                            ::testing::ValuesIn(ov::test::conformance::generate_configs(CommonTestUtils::DEVICE_MULTI))),
                    InferRequestPreprocessDynamicallyInSetBlobTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, InferRequestPreprocessConversionTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(netPrecisionsPreprocess),
                                 ::testing::ValuesIn(ioPrecisionsPreprocess),
                                 ::testing::ValuesIn(ioPrecisionsPreprocess),
                                 ::testing::ValuesIn(netLayoutsPreprocess),
                                 ::testing::ValuesIn(ioLayoutsPreprocess),
                                 ::testing::ValuesIn(ioLayoutsPreprocess),
                                 ::testing::Bool(),
                                 ::testing::Bool(),
                                 ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                 ::testing::ValuesIn(ov::test::conformance::generate_configs(CommonTestUtils::DEVICE_AUTO))),
                         InferRequestPreprocessConversionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, InferRequestPreprocessDynamicallyInSetBlobTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(netPrecisionsPreprocess),
                                 ::testing::Bool(),
                                 ::testing::Bool(),
                                 ::testing::ValuesIn(netLayoutsPreprocess),
                                 ::testing::Bool(),
                                 ::testing::Bool(),
                                 ::testing::Values(true), // only SetBlob
                                 ::testing::Values(true), // only SetBlob
                                 ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                 ::testing::ValuesIn(ov::test::conformance::generate_configs(CommonTestUtils::DEVICE_AUTO))),
                         InferRequestPreprocessDynamicallyInSetBlobTest::getTestCaseName);
}  // namespace
