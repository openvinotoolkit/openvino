// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/common_utils.hpp"
#include "snippets/add.hpp"
#include "subgraph_simple.hpp"
#include "ngraph_functions/builders.hpp"
#include "functional_test_utils/skip_tests_config.hpp"

namespace ov {
namespace test {
namespace snippets {

std::string Add::getTestCaseName(testing::TestParamInfo<ov::test::snippets::AddParams> obj) {
    ov::Shape inputShapes0, inputShapes1, newInputShapes;
    std::string targetDevice;
    size_t num_nodes, num_subgraphs;
    std::tie(inputShapes0, inputShapes1, num_nodes, num_subgraphs, targetDevice) = obj.param;

    std::ostringstream result;
    result << "IS[0]=" << CommonTestUtils::vec2str(inputShapes0) << "_";
    result << "IS[1]=" << CommonTestUtils::vec2str(inputShapes1) << "_";
    result << "#N=" << num_nodes << "_";
    result << "#S=" << num_subgraphs << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void Add::SetUp() {
    ov::Shape inputShape0, inputShape1;
    std::tie(inputShape0, inputShape1, ref_num_nodes, ref_num_subgraphs, targetDevice) = this->GetParam();
    init_input_shapes({{{}, {inputShape0, }}, {{}, {inputShape1, }}});

    auto f = ov::test::snippets::AddFunction({inputShape0, inputShape1});
    function = f.getOriginal();
}

void AddSinh::SetUp() {
    ov::Shape inputShape0, inputShape1;
    std::tie(inputShape0, inputShape1, ref_num_nodes, ref_num_subgraphs, targetDevice) = this->GetParam();
    init_input_shapes({{{}, {inputShape0, }}, {{}, {inputShape1, }}});

    auto f = ov::test::snippets::AddSinhFunction({inputShape0, inputShape1});
    function = f.getOriginal();
}

std::string AddSinhConst::getTestCaseName(testing::TestParamInfo<ov::test::snippets::AddConstParams> obj) {
    ov::Shape inputShapes, newInputShapes;
    std::string targetDevice;
    size_t num_nodes, num_subgraphs;
    std::tie(inputShapes, num_nodes, num_subgraphs, targetDevice) = obj.param;

    std::ostringstream result;
    result << "IS[0]=" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "#N=" << num_nodes << "_";
    result << "#S=" << num_subgraphs << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void AddSinhConst::SetUp() {
    ov::Shape inputShape;
    std::tie(inputShape, ref_num_nodes, ref_num_subgraphs, targetDevice) = this->GetParam();
    init_input_shapes({{{}, {inputShape, }}});

    auto f = ov::test::snippets::AddSinhConstFunction({inputShape});
    function = f.getOriginal();
}

std::string AddSinhPair::getTestCaseName(testing::TestParamInfo<ov::test::snippets::AddParamsPair> obj) {
    std::vector<ov::Shape> input_shapes;
    std::string targetDevice;
    size_t num_nodes, num_subgraphs;
    std::tie(input_shapes, num_nodes, num_subgraphs, targetDevice) = obj.param;
    if (input_shapes.size() != 2)
        IE_THROW() << "Invalid input shapes vector size";
    std::ostringstream result;
    result << "IS[0]=" << CommonTestUtils::vec2str(input_shapes[0]) << "_";
    result << "IS[1]=" << CommonTestUtils::vec2str(input_shapes[1]) << "_";
    result << "#N=" << num_nodes << "_";
    result << "#S=" << num_subgraphs << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void AddSinhPair::SetUp() {
    std::vector<ov::Shape> input_shapes;
    std::tie(input_shapes, ref_num_nodes, ref_num_subgraphs, targetDevice) = this->GetParam();
    std::vector<InputShape> is;
    for (const auto& s : input_shapes) {
        is.emplace_back(InputShape {{}, {s, }});
    }
    init_input_shapes(is);
    auto f = ov::test::snippets::AddSinhFunction({input_shapes[0], input_shapes[1]});
    function = f.getOriginal();
}

TEST_P(Add, CompareWithRefImpl) {
    run();
    validateNumSubgraphs();
}

TEST_P(AddSinh, CompareWithRefImpl) {
    run();
    validateNumSubgraphs();
}

TEST_P(AddSinhConst, CompareWithRefImpl) {
    run();
    validateNumSubgraphs();
}

TEST_P(AddSinhPair, CompareWithRefImpl) {
    run();
    validateNumSubgraphs();
}

} // namespace snippets
} // namespace test
} // namespace ov
