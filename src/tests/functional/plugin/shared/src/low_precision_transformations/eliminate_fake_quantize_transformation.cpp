// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/eliminate_fake_quantize_transformation.hpp"

#include <tuple>
#include <sstream>
#include <string>
#include <vector>

#include <transformations/init_node_info.hpp>
#include "openvino/util/common_util.hpp"
#include "ov_lpt_models/fuse_fake_quantize.hpp"

namespace LayerTestsDefinitions {

std::string EliminateFakeQuantizeTransformation::getTestCaseName(const testing::TestParamInfo<EliminateFakeQuantizeTransformationParams>& obj) {
    std::string targetDevice;
    EliminateFakeQuantizeTransformationTestValues testValues;
    std::tie(targetDevice, testValues) = obj.param;

    std::ostringstream result;
    result << targetDevice << "_" <<
        testValues.actual.precisionBefore << "_" <<
        testValues.actual.fakeQuantizeOnData1 << "_" <<
        testValues.actual.fakeQuantizeOnData2;
    return result.str();
}

void EliminateFakeQuantizeTransformation::SetUp() {
    EliminateFakeQuantizeTransformationTestValues testValues;
    std::tie(targetDevice, testValues) = this->GetParam();

    // Convolution is used in a model as operation with specific precision requirements on data branch
    // to test the transformation place in LPT pipeline:
    // markup transformations and FakeQuantize operation decomposition transformation have to handle FakeQuantize as usual
    function = ngraph::builder::subgraph::FuseFakeQuantizeFunction::get(
        testValues.inputShape,
        testValues.actual.precisionBefore,
        testValues.actual.fakeQuantizeOnData1,
        testValues.actual.fakeQuantizeOnData2,
        {});

    ov::pass::InitNodeInfo().run_on_model(function);
}

TEST_P(EliminateFakeQuantizeTransformation, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    Run();

    EliminateFakeQuantizeTransformationTestValues testValues;
    std::tie(targetDevice, testValues) = this->GetParam();

    const auto& rtInfo = LayerTestsCommon::getRuntimeInfo();

    auto exist = testValues.expected.exist;
    auto absent = testValues.expected.absent;
    auto int8_convolutions = 0ull;
    for (const auto& it : rtInfo) {
        const auto& nameIt = it.second.find("originalLayersNames");
        const auto& fused_name = nameIt->second.as<std::string>();

        const auto names = ov::util::split(fused_name, ',', true);
        for (const auto& name : names) {
            ASSERT_TRUE(absent.find(name) == absent.end());
            exist.erase(name);
        }

        const auto& type_it = it.second.find("layerType");
        const auto& type = type_it->second.as<std::string>();

        if (type == "Convolution") {
            const auto& precision_it = it.second.find("runtimePrecision");
            const auto& precision = precision_it->second.as<std::string>();
            if (precision == "U8") {
                int8_convolutions++;
            }
        }
    }

    ASSERT_TRUE(exist.empty());
    ASSERT_EQ(testValues.expected.int8_convolutions, int8_convolutions);
};

}  // namespace LayerTestsDefinitions
