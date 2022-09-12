// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/common_utils.hpp"
#include "snippets/add.hpp"
#include "subgraph_simple.hpp"
#include "ngraph_functions/builders.hpp"

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

std::string AddSinhDynamic::getTestCaseName(testing::TestParamInfo<ov::test::snippets::AddDynamicParams> obj) {
    InputShape inputShape1, inputShape2;
    std::string targetDevice;
    size_t num_nodes, num_subgraphs;
    std::tie(inputShape1, inputShape2, num_nodes, num_subgraphs, targetDevice) = obj.param;

    std::ostringstream result;
    result << "IS[0]=" << CommonTestUtils::partialShape2str({inputShape1.first}) << "_";
    result << "IS[1]=" << CommonTestUtils::partialShape2str({inputShape2.first}) << "_";
    result << "TS[0]=";
    for (const auto& item : inputShape1.second)
        result << CommonTestUtils::vec2str(item) << "_";
    result << "TS[1]=";
    for (const auto& item : inputShape2.second)
        result << CommonTestUtils::vec2str(item) << "_";
    result << "#N=" << num_nodes << "_";
    result << "#S=" << num_subgraphs << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void AddSinhDynamic::SetUp() {
    InputShape inputShape1, inputShape2;
    std::tie(inputShape1, inputShape2, ref_num_nodes, ref_num_subgraphs, targetDevice) = this->GetParam();
    init_input_shapes({inputShape1, inputShape2});

    auto f = ov::test::snippets::AddSinhFunction({inputShape1.first, inputShape2.first});
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

TEST_P(AddSinhDynamic, CompareWithRefImpl) {
    enableSnippetsDynamismSupport();
    run();
    validateNumSubgraphs();
}

} // namespace snippets
} // namespace test
} // namespace ov
