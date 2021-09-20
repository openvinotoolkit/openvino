// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/add_branch_selection_transformation.hpp"

#include <memory>
#include <tuple>

#include <transformations/init_node_info.hpp>
#include "lpt_ngraph_functions/add_function.hpp"

namespace LayerTestsDefinitions {

std::string AddBranchSelectionTransformation::getTestCaseName(const testing::TestParamInfo<AddBranchSelectionTransformationParams>& obj) {
    ngraph::element::Type netPrecision;
    ngraph::PartialShape inputShapes;
    std::string targetDevice;
    auto params = LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8();
    AddBranchSelectionTestValues param;
    std::tie(netPrecision, inputShapes, targetDevice, param) = obj.param;

    std::ostringstream result;
    result << getTestCaseNameByParams(netPrecision, inputShapes, targetDevice, params);

    auto toString = [](const ngraph::builder::subgraph::FakeQuantizeOnData& fqOnData) -> std::string {
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

void AddBranchSelectionTransformation::SetUp() {
    ngraph::element::Type precision;
    ngraph::PartialShape inputShape;
    AddBranchSelectionTestValues param;
    std::tie(precision, inputShape, targetDevice, param) = this->GetParam();

    function = ngraph::builder::subgraph::AddFunction::getOriginalSubgraphWithConvolutions(
        precision,
        inputShape,
        false,
        param.branch1.fakeQuantizeBefore,
        param.branch1.convolution,
        param.branch1.fakeQuantizeAfter,
        param.branch2.fakeQuantizeBefore,
        param.branch2.convolution,
        param.branch2.fakeQuantizeAfter,
        param.fakeQuantizeAfter);

    ngraph::pass::InitNodeInfo().run_on_function(function);
}

TEST_P(AddBranchSelectionTransformation, CompareWithRefImpl) {
    Run();

    const auto params = std::get<3>(GetParam());
    std::vector<std::pair<std::string, std::string>> expectedReorders = params.expectedReorders;

    auto rtInfo = LayerTestsCommon::getRuntimeInfo();
    for (auto it : rtInfo) {
        const auto& typeIt = it.second.find("layerType");
        const auto type = ngraph::as_type_ptr<ngraph::VariantWrapper<std::string>>(typeIt->second)->get();
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
            const auto precision = ngraph::as_type_ptr<ngraph::VariantWrapper<std::string>>(precisionIt->second)->get();
            ASSERT_EQ("U8", precision);
        }
    }

    ASSERT_TRUE(expectedReorders.empty()) << "Some Reorder operations were not found in execution graph";
};

}  // namespace LayerTestsDefinitions
