// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/fake_quantize_decomposition_test.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include "ov_ops/type_relaxed.hpp"
#include "fake_quantize_helper.hpp"
#include "function_helper.hpp"

namespace ov {
namespace test {
namespace snippets {
std::string FakeQuantizeDecompositionTest::getTestCaseName(const testing::TestParamInfo<testsParams>& obj) {
    std::ostringstream result;
    const auto values = std::get<0>(obj.param);
    const auto operation = std::get<1>(obj.param);
    const auto operations_number = std::get<2>(obj.param);
    const auto targetDevice = std::get<3>(obj.param);

    const auto type_info = operation.first->get_type_info();
    const auto operationString = ov::is_type<ov::op::v0::Parameter>(operation.first) ?
        "nullptr" :
        (std::string(type_info.name) + "_" + std::string(type_info.version_id));

    result << "IS=" << ov::test::utils::vec2str(values.inputShape) << "_";
    result << "netPRC=" << values.modelType << "_";
    result << "D=" << targetDevice << "_";
    result << "IN=" << values.inputType << "_";
    result << "OP=" << operationString << "_";
    result << "ON1=" << std::string(operation.second.first) << "_";
    result << "ON1=" << std::string(operation.second.second) << "_";
    result << "LP=" << values.zeroPoint;
    result << "SH1=" << values.fakeQuantizeShapes[0] << "SH2=" << values.fakeQuantizeShapes[1]
           << "SH3=" << values.fakeQuantizeShapes[2] << "SH4=" << values.fakeQuantizeShapes[3];
    return result.str();
}

void FakeQuantizeDecompositionTest::SetUp() {
    auto& testsParams = this->GetParam();

    const auto values = std::get<0>(testsParams);
    const auto operation = std::get<1>(testsParams);
    const auto operations_number = std::get<2>(testsParams);
    targetDevice = std::get<3>(testsParams);

    ref_num_nodes = operations_number.first;
    ref_num_subgraphs = operations_number.second;

    init_input_shapes({{values.inputShape, {values.inputShape}}});

    std::shared_ptr<ov::Node> op = ov::is_type<ov::op::v0::Parameter>(operation.first) ? nullptr : operation.first;
    function = ov::test::snippets::FakeQuantizeFunction::getOperationAndFakeQuantize(
        {values.inputShape},
        values.inputType,
        values.fakeQuantizeShapes,
        values.zeroPoint,
        {},
        op);

    expected_layer_type = std::string(operation.second.first);
    expected_original_layers_names = operation.second.second;
}

void FakeQuantizeDecompositionTest::validate() {
    SnippetsTestsCommon::validate();
    validateOriginalLayersNamesByType(expected_layer_type, expected_original_layers_names);
}

void FakeQuantizeDecompositionTest::validateOriginalLayersNamesByType(const std::string& layerType,
                                                                       const std::string& originalLayersNames) {
    const auto& compiled_model = compiledModel.get_runtime_model();
    for (const auto& op : compiled_model->get_ops()) {
        const auto& rtInfo = op->get_rt_info();

        const auto& typeIt = rtInfo.find("layerType");
        if (typeIt == rtInfo.end())
            continue;
        const auto type = typeIt->second.as<std::string>();
        if (type == layerType) {
            const auto& nameIt = rtInfo.find("originalLayersNames");
            ASSERT_NE(nameIt, rtInfo.end()) << "Failed to find originalLayersNames in " << op->get_friendly_name() << " rt_info.";
            const auto name = nameIt->second.as<std::string>();
            ASSERT_EQ(originalLayersNames, name);
            return;
        }
    }

    ASSERT_TRUE(false) << "Layer type '" << layerType << "' was not found in compiled model";
}

TEST_P(FakeQuantizeDecompositionTest, CompareWithRefImpl) {
    run();
};
}  // namespace snippets
}  // namespace test
}  // namespace ov
