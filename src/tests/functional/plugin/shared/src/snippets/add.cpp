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
    ov::element::Type type;
    std::string targetDevice;
    size_t num_nodes, num_subgraphs;
    std::tie(inputShapes0, inputShapes1, type, num_nodes, num_subgraphs, targetDevice) = obj.param;

    std::ostringstream result;
    result << "IS[0]=" << ov::test::utils::vec2str(inputShapes0) << "_";
    result << "IS[1]=" << ov::test::utils::vec2str(inputShapes1) << "_";
    result << "T=" << type << "_";
    result << "#N=" << num_nodes << "_";
    result << "#S=" << num_subgraphs << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void Add::SetUp() {
    ov::Shape inputShape0, inputShape1;
    ov::element::Type type;
    std::tie(inputShape0, inputShape1, type, ref_num_nodes, ref_num_subgraphs, targetDevice) = this->GetParam();
    init_input_shapes({{{}, {inputShape0, }}, {{}, {inputShape1, }}});

    auto f = ov::test::snippets::AddFunction({inputShape0, inputShape1});
    function = f.getOriginal();
    setInferenceType(type);
}

std::string AddConst::getTestCaseName(testing::TestParamInfo<ov::test::snippets::AddConstParams> obj) {
    ov::Shape inputShapes, newInputShapes;
    ov::element::Type type;
    std::string targetDevice;
    size_t num_nodes, num_subgraphs;
    std::tie(inputShapes, type, num_nodes, num_subgraphs, targetDevice) = obj.param;

    std::ostringstream result;
    result << "IS[0]=" << ov::test::utils::vec2str(inputShapes) << "_";
    result << "T=" << type << "_";
    result << "#N=" << num_nodes << "_";
    result << "#S=" << num_subgraphs << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void AddConst::SetUp() {
    ov::Shape inputShape;
    ov::element::Type type;
    std::tie(inputShape, type, ref_num_nodes, ref_num_subgraphs, targetDevice) = this->GetParam();
    init_input_shapes({{{}, {inputShape, }}});

    auto f = ov::test::snippets::AddConstFunction({inputShape});
    function = f.getOriginal();
    setInferenceType(type);
}

void AddRollConst::SetUp() {
    ov::Shape inputShape;
    ov::element::Type type;
    std::tie(inputShape, type, ref_num_nodes, ref_num_subgraphs, targetDevice) = this->GetParam();
    init_input_shapes({{{}, {inputShape, }}});

    auto f = ov::test::snippets::AddRollConstFunction({inputShape});
    function = f.getOriginal();
    setInferenceType(type);
}

std::string AddPair::getTestCaseName(testing::TestParamInfo<ov::test::snippets::AddParamsPair> obj) {
    std::vector<ov::Shape> input_shapes;
    ov::element::Type type;
    std::string targetDevice;
    size_t num_nodes, num_subgraphs;
    std::tie(input_shapes, type, num_nodes, num_subgraphs, targetDevice) = obj.param;
    if (input_shapes.size() != 2)
        IE_THROW() << "Invalid input shapes vector size";
    std::ostringstream result;
    result << "IS[0]=" << ov::test::utils::vec2str(input_shapes[0]) << "_";
    result << "IS[1]=" << ov::test::utils::vec2str(input_shapes[1]) << "_";
    result << "T=" << type << "_";
    result << "#N=" << num_nodes << "_";
    result << "#S=" << num_subgraphs << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void AddPair::SetUp() {
    std::vector<ov::Shape> input_shapes;
    ov::element::Type type;
    std::tie(input_shapes, type, ref_num_nodes, ref_num_subgraphs, targetDevice) = this->GetParam();
    std::vector<InputShape> is;
    for (const auto& s : input_shapes) {
        is.emplace_back(InputShape {{}, {s, }});
    }
    init_input_shapes(is);
    auto f = ov::test::snippets::AddFunction({input_shapes[0], input_shapes[1]});
    function = f.getOriginal();
    setInferenceType(type);
}

TEST_P(Add, CompareWithRefImpl) {
    run();
    validateNumSubgraphs();
}

TEST_P(AddConst, CompareWithRefImpl) {
    run();
    validateNumSubgraphs();
}

TEST_P(AddRollConst, CompareWithRefImpl) {
    run();
    validateNumSubgraphs();
}

TEST_P(AddPair, CompareWithRefImpl) {
    run();
    validateNumSubgraphs();
}

} // namespace snippets
} // namespace test
} // namespace ov
