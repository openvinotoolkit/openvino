
// Copyright (C) 2022 Intel Corporation
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
std::string FakeQuantizeDecompositionTest::getTestCaseName(testing::TestParamInfo<testsParams> obj) {
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
}

TEST_P(FakeQuantizeDecompositionTest, CompareWithRefImpl) {
    run();

    const auto operation = std::get<1>(this->GetParam());
    auto elementType = std::string(operation.second.first);
    validateOriginalLayersNamesByType(elementType, operation.second.second);

    validateNumSubgraphs();
};
}  // namespace snippets
}  // namespace test
}  // namespace ov
