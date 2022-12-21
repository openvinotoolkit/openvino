// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "pass/softmax_decomposition.hpp"
#include "common_test_utils/common_utils.hpp"
#include "subgraph_softmax.hpp"
#include "subgraph_lowered.hpp"

#include "snippets/pass/softmax_decomposition.hpp"
#include "snippets/pass/insert_load_store.hpp"
#include "snippets/pass/insert_movebroadcast.hpp"
#include "snippets/pass/insert_buffer.hpp"
#include "snippets/pass/convert_power_to_powerstatic.hpp"


namespace ov {
namespace test {
namespace snippets {

std::string SoftmaxTests::getTestCaseName(testing::TestParamInfo<SoftmaxParams> obj) {
    Shape inputShape;
    int axis;
    std::tie(inputShape, axis) = obj.param;
    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShape) << "_";
    result << "Axis=" << axis << "_";
    return result.str();
}

void SoftmaxTests::SetUp() {
    LoweringTests::SetUp();

    const size_t count = 10;
    manager.register_pass<ngraph::snippets::pass::SoftmaxDecomposition>(count);
    manager.register_pass<ngraph::snippets::pass::ConvertPowerToPowerStatic>();
    manager.register_pass<ngraph::snippets::pass::InsertLoad>(count);
    manager.register_pass<ngraph::snippets::pass::InsertStore>(count);
    manager.register_pass<ngraph::snippets::pass::InsertMoveBroadcast>();
    Shape inputShape;
    int axis;
    std::tie(inputShape, axis) = this->GetParam();
    snippets_function = std::make_shared<SoftmaxLoweredFunction>(std::vector<PartialShape>{inputShape}, axis);
    master_shape = inputShape;
}

std::string AddSoftmaxTests::getTestCaseName(testing::TestParamInfo<AddSoftmaxParams> obj) {
    Shape inputShape0, inputShape1;
    int axis;
    std::tie(inputShape0, inputShape1, axis) = obj.param;
    std::ostringstream result;
    result << "IS[0]=" << CommonTestUtils::vec2str(inputShape0) << "_";
    result << "IS[1]=" << CommonTestUtils::vec2str(inputShape1) << "_";
    result << "Axis=" << axis << "_";
    return result.str();
}

void AddSoftmaxTests::SetUp() {
    LoweringTests::SetUp();

    const size_t count = 10;
    manager.register_pass<ngraph::snippets::pass::InsertBuffer>();
    manager.register_pass<ngraph::snippets::pass::SoftmaxDecomposition>(count);
    manager.register_pass<ngraph::snippets::pass::ConvertPowerToPowerStatic>();
    manager.register_pass<ngraph::snippets::pass::InsertLoad>(count);
    manager.register_pass<ngraph::snippets::pass::InsertStore>(count);
    manager.register_pass<ngraph::snippets::pass::InsertMoveBroadcast>();
    Shape inputShape0, inputShape1;
    int axis;
    std::tie(inputShape0, inputShape1, axis) = this->GetParam();
    snippets_function = std::make_shared<AddSoftmaxLoweredFunction>(std::vector<PartialShape>{inputShape0, inputShape1}, axis);

    ov::PartialShape master_pshape(inputShape0);
    ov::PartialShape::broadcast_merge_into(master_pshape, inputShape1, op::AutoBroadcastType::NUMPY);
    master_shape = master_pshape.get_shape();
}

TEST_P(SoftmaxTests, SoftmaxDecomposition) {
    PartialShape scheduler_shape({master_shape[master_shape.size() - 2],
                                  master_shape[master_shape.size() - 1]});
    auto subgraph = getLoweredSubgraph(snippets_function->getOriginal(), scheduler_shape);
    function = subgraph->get_body();
    function_ref = snippets_function->getLowered();
}

TEST_P(AddSoftmaxTests, AddSoftmaxDecomposition) {
    PartialShape scheduler_shape({master_shape[master_shape.size() - 2],
                                  master_shape[master_shape.size() - 1]});
    auto subgraph = getLoweredSubgraph(snippets_function->getOriginal(), scheduler_shape);
    function = subgraph->get_body();
    function_ref = snippets_function->getLowered();
}

namespace SoftmaxTestsInstantiation {
std::vector<ov::Shape> inputShape{{12, 4, 12, 12, 127}, {12, 4, 12, 12, 1}};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_SoftmaxDecomposition, SoftmaxTests,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inputShape),
                                 ::testing::Values(-1)),
                         SoftmaxTests::getTestCaseName);

}  // namespace SoftmaxTestsInstantiation

namespace AddSoftmaxTestsInstantiation {
std::vector<ov::Shape> inputShape0{{12, 4, 12, 12, 17}, {12, 4, 12, 12, 1}};
std::vector<ov::Shape> inputShape1{{12, 4, 12, 12, 17}, {12, 4, 12, 12, 1}};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_AddSoftmaxDecomposition, AddSoftmaxTests,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inputShape0),
                                 ::testing::ValuesIn(inputShape1),
                                 ::testing::Values(-1)),
                         AddSoftmaxTests::getTestCaseName);

}  // namespace AddSoftmaxTestsInstantiation

}  // namespace snippets
}  // namespace test
}  // namespace ov
