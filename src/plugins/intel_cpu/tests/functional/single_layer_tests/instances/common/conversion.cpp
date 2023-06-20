// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/classes/conversion.hpp"
#include "shared_test_classes/single_layer/conversion.hpp"
#include "test_utils/cpu_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ngraph::helpers;
using namespace ov::test;

namespace CPULayerTestsDefinitions {
namespace Conversion {

std::vector<CPUSpecificParams> memForm4D_dynamic = {
    CPUSpecificParams({nchw}, {nchw}, {}, "acl"),
    CPUSpecificParams({nhwc}, {nhwc}, {}, "acl"),
};

INSTANTIATE_TEST_SUITE_P(smoke_ConvertCPULayerTest_Dynamic, ConvertCPULayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inShapes_4D_dynamic()),
                                ::testing::ValuesIn(precisions()),
                                ::testing::ValuesIn(precisions()),
                                ::testing::ValuesIn(memForm4D_dynamic)),
                        ConvertCPULayerTest::getTestCaseName);

std::vector<CPUSpecificParams> memForm4D_static_common = {
    CPUSpecificParams({nchw}, {nchw}, {}, {}),
    CPUSpecificParams({nhwc}, {nhwc}, {}, {}),
};

INSTANTIATE_TEST_SUITE_P(smoke_ConvertCPULayerTest, ConvertCPULayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inShapes_4D_static()),
                                ::testing::ValuesIn(precisions()),
                                ::testing::ValuesIn(precisions()),
                                ::testing::ValuesIn(memForm4D_static_common)),
                        ConvertCPULayerTest::getTestCaseName);

}  // namespace Conversion
}  // namespace CPULayerTestsDefinitions