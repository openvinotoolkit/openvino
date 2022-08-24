// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "pass/insert_floor_after_int_div.hpp"
#include "common_test_utils/common_utils.hpp"
#include <subgraph_lowered.hpp>

namespace ov {
namespace test {
namespace snippets {

std::string InsertFloorAfterIntDivTests::getTestCaseName(testing::TestParamInfo<insertFloorAfterIntDivParams> obj) {
    Shape inputShape;
    bool is_pythondiv;
    element::Type inputType;
    std::tie(inputShape, is_pythondiv, inputType) = obj.param;
    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShape) << "_";
    result << "PythonDiv=" << is_pythondiv << "_";
    result << "IT=" << inputType << "_";
    return result.str();
}

void InsertFloorAfterIntDivTests::SetUp() {
    TransformationTestsF::SetUp();
    Shape inputShape;
    bool is_pythondiv;
    element::Type inputType;
    std::tie(inputShape, is_pythondiv, inputType) = this->GetParam();
    std::vector<Shape> inputShapes = {inputShape, inputShape};
    snippets_function = std::make_shared<DivFunctionLoweredFunction>(inputShapes, is_pythondiv, inputType);
}

TEST_P(InsertFloorAfterIntDivTests, InsertFloorAfterIntDiv) {
    auto subgraph = getLoweredSubgraph(snippets_function->getOriginal());
    function = subgraph->get_body();
    function_ref = snippets_function->getLowered();
}

namespace InsertFloorAfterIntDivInstantiation {

std::vector<element::Type> inputType {
    element::f32,
    element::i32,
    element::bf16,
    element::i8,
    element::u8
};

std::vector<bool> pythondiv = { true, false };

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_InsertFloorAfterIntDivTests, InsertFloorAfterIntDivTests,
                         ::testing::Combine(
                                 ::testing::Values(Shape {3, 2, 16, 16}),
                                 ::testing::ValuesIn(pythondiv),
                                 ::testing::ValuesIn(inputType)),
                         InsertFloorAfterIntDivTests::getTestCaseName);

}  // namespace InsertFloorAfterIntDivInstantiation
}  // namespace snippets
}  // namespace test
}  // namespace ov
