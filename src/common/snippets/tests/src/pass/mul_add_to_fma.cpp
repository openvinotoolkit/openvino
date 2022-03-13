// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <pass/mul_add_to_fma.hpp>
#include <gtest/gtest.h>
#include <subgraph_simple.hpp>
#include "common_test_utils/common_utils.hpp"
#include "snippets/pass/mul_add_to_fma.hpp"

namespace ov {
namespace test {
namespace snippets {

std::string MulAddToFMATests::getTestCaseName(testing::TestParamInfo<MulAddToFMAParams> obj) {
    std::vector<PartialShape> inputShapes(3);
    size_t add_input_idx;
    bool constant_input;
    std::tie(inputShapes[0], inputShapes[1], inputShapes[2], add_input_idx, constant_input) = obj.param;

    std::ostringstream result;
    for (size_t i = 0; i < inputShapes.size(); i++)
        result << "IS[" << i << "]=" << inputShapes[i] << "_";
    result << "add_input_idx=" << add_input_idx << (constant_input ? "_constant_input" : "");
    return result.str();
}

void MulAddToFMATests::SetUp() {
    TransformationTestsF::SetUp();
    std::vector<PartialShape> inputShapes(3);
    size_t add_input_idx;
    bool constant_input;
    std::tie(inputShapes[0], inputShapes[1], inputShapes[2], add_input_idx, constant_input) = this->GetParam();
    snippets_function = std::make_shared<EltwiseWithMulAddFunction>(inputShapes, add_input_idx, constant_input);

    manager.register_pass<ngraph::snippets::pass::MulAddToFMA>();
}

TEST_P(MulAddToFMATests, AddBroadcast) {
    model = snippets_function->getOriginal();
    model_ref = snippets_function->getReference();
}

namespace MulAddToFMATestsInstantiation {
std::vector<PartialShape> in_shapes_0 = {{1, 3, 2, 2}};
std::vector<PartialShape> in_shapes_1 = {{1, 3, 2, 2}};
std::vector<PartialShape> in_shapes_2 = {{1, 3, 2, 2}, {}};
std::vector<size_t> in_idxes_for_add = {0, 1};
std::vector<bool> constant_input = {false, true};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets, MulAddToFMATests,
                        ::testing::Combine(
                                ::testing::ValuesIn(in_shapes_0),
                                ::testing::ValuesIn(in_shapes_1),
                                ::testing::ValuesIn(in_shapes_2),
                                ::testing::ValuesIn(in_idxes_for_add),
                                ::testing::ValuesIn(constant_input)),
                        MulAddToFMATests::getTestCaseName);

} // namespace MulAddToFMATestsInstantiation

TEST_F(TransformationTestsF, smoke_Snippets_MulAddToFMANegative) {
    auto data0 = std::make_shared<op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 3, 2, 2});
    auto data1 = std::make_shared<op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 3, 2, 2});
    auto data2 = std::make_shared<op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 3, 2, 2});

    auto mul = std::make_shared<op::v1::Multiply>(data0, data1);
    auto additional_consumer = std::make_shared<op::v0::Relu>(mul);
    auto add = std::make_shared<op::v1::Add>(mul, data2);

    model = std::make_shared<Model>(ov::NodeVector{add, additional_consumer}, ov::ParameterVector{data0, data1, data2});
}

}  // namespace snippets
}  // namespace test
}  // namespace ov