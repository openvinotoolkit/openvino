// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/preprocessing.hpp"

#ifdef ENABLE_GAPI_PREPROCESSING

namespace ov {
namespace test {
namespace behavior {

const std::vector<ov::element::Type> inputPrecisions = {ov::element::u16, ov::element::f32};

const std::vector<ov::AnyMap> configs = {{}};

INSTANTIATE_TEST_SUITE_P(BehaviourPreprocessingTestsViaSetTensor,
                         PreprocessingPrecisionConvertTests,
                         ::testing::Combine(::testing::ValuesIn(inputPrecisions),
                                            ::testing::Values(1, 2, 3, 4, 5),  // Number of input tensor channels
                                            ::testing::Values(true),           // Use SetInput
                                            ::testing::Values(ov::test::utils::DEVICE_CPU),
                                            ::testing::ValuesIn(configs)),
                         PreprocessingPrecisionConvertTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(BehaviourPreprocessingTestsViaGetTensor,
                         PreprocessingPrecisionConvertTests,
                         ::testing::Combine(::testing::ValuesIn(inputPrecisions),
                                            ::testing::Values(4, 5),  // Number of input tensor channels (blob_copy only
                                                                      // supports 4d and 5d tensors)
                                            ::testing::Values(false),  // use GetBlob
                                            ::testing::Values(ov::test::utils::DEVICE_CPU),
                                            ::testing::ValuesIn(configs)),
                         PreprocessingPrecisionConvertTests::getTestCaseName);
}  // namespace behavior
}  // namespace test
}  // namespace ov

#endif  // ENABLE_GAPI_PREPROCESSING
