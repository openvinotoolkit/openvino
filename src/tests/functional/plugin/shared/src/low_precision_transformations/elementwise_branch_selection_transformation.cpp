// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/elementwise_branch_selection_transformation.hpp"

#include <memory>
#include <tuple>

#include "transformations/init_node_info.hpp"
#include "ov_lpt_models/add.hpp"

namespace LayerTestsDefinitions {

std::string ElementwiseBranchSelectionTransformation::getTestCaseName(const testing::TestParamInfo<ElementwiseBranchSelectionTransformationParams>& obj) {
    ov::element::Type netPrecision;
    ov::PartialShape inputShapes;
    std::string targetDevice;
    auto params = LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8();
    ElementwiseBranchSelectionTestValues param;
    std::string elementwiseType;
    std::tie(netPrecision, inputShapes, targetDevice, param, elementwiseType) = obj.param;

    std::ostringstream result;
    result << get_test_case_name_by_params(netPrecision, inputShapes, targetDevice, params) <<
           "_elementwiseType_" << elementwiseType;

    auto toString = [](const ov::builder::subgraph::FakeQuantizeOnData& fqOnData) -> std::string {
        if (fqOnData.empty()) {
            return "";
        }

        std::stringstream ss;
        ss << "_on_branch1_" <<
            fqOnData.inputLowValues[0] << "_" <<
            fqOnData.inputHighValues[0] << "_" <<
            fqOnData.outputLowValues[0] << "_" <<
            fqOnData.outputHighValues[0];
        return ss.str();
    };

    result <<
        "_on_branch1_" << toString(param.branch1.fakeQuantizeBefore) << toString(param.branch1.fakeQuantizeAfter) <<
        "_on_branch1_" << toString(param.branch1.fakeQuantizeBefore) << toString(param.branch1.fakeQuantizeAfter) <<
        "_" << toString(param.fakeQuantizeAfter);

    return result.str();
}

void ElementwiseBranchSelectionTransformation::SetUp() {
    ov::element::Type precision;
    ov::PartialShape inputShape;
    ElementwiseBranchSelectionTestValues param;
    std::string elementwiseType;
    std::tie(precision, inputShape, targetDevice, param, elementwiseType) = this->GetParam();

    init_input_shapes({ inputShape, inputShape });

    function = ov::builder::subgraph::AddFunction::getOriginalSubgraphWithConvolutions(
        precision,
        inputShape,
        false,
        elementwiseType,
        param.branch1.fakeQuantizeBefore,
        param.branch1.convolution,
        param.branch1.fakeQuantizeAfter,
        param.branch2.fakeQuantizeBefore,
        param.branch2.convolution,
        param.branch2.fakeQuantizeAfter,
        param.fakeQuantizeAfter);

    ov::pass::InitNodeInfo().run_on_model(function);
}

void ElementwiseBranchSelectionTransformation::run() {
    LayerTransformation::run();

    const auto params = std::get<3>(GetParam());
    const auto elementwiseType = std::get<4>(GetParam());

    std::vector<std::pair<std::string, std::string>> expectedReorders = params.expectedReorders;
    if (!expectedReorders.empty()) {
        auto rtInfo = LayerTransformation::get_runtime_info();
        for (auto it : rtInfo) {
            const auto& typeIt = it.second.find("layerType");
            const auto type = typeIt->second.as<std::string>();
            if (type == "Reorder") {
                const auto name = it.first;
                bool wasFound = false;
                for (auto it = expectedReorders.begin(); it != expectedReorders.end(); ++it) {
                    auto pair = *it;
                    const std::string parent = name.substr(0, name.find("_"));
                    const std::string child = name.substr(name.rfind("_") + 1, name.size() - name.rfind("_") - 1);
                    if ((pair.first == parent) && (pair.second == child)) {
                        expectedReorders.erase(it);
                        wasFound = true;
                        break;
                    }
                }

                ASSERT_TRUE(wasFound) << it.first << " was not found in expected list";
            } else if (type == "Convolution") {
                const auto& precisionIt = it.second.find("runtimePrecision");
                const auto precision = precisionIt->second.as<std::string>();
                ASSERT_EQ("u8", precision);
            }
        }

        ASSERT_TRUE(expectedReorders.empty()) << "Some Reorder operations were not found in execution graph";
    }

    for (auto it : params.expectedPrecisions) {
        const auto actualPrecision = get_runtime_precision_by_fused_name(
                it.first == "eltwise" ? elementwiseType : it.first);
        ASSERT_EQ(it.second, actualPrecision) << "actual precision for operation '" << it.first << "' is not correct";
    }
}

TEST_P(ElementwiseBranchSelectionTransformation, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
};

}  // namespace LayerTestsDefinitions
