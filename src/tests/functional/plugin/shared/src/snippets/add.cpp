// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/common_utils.hpp"
#include "snippets/add.hpp"
#include "subgraph_simple.hpp"

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
    result << "IS[0]=" << CommonTestUtils::vec2str(inputShapes0) << "_";
    result << "IS[1]=" << CommonTestUtils::vec2str(inputShapes1) << "_";
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

void AddSinh::SetUp() {
    ov::Shape inputShape0, inputShape1;
    ov::element::Type type;
    std::tie(inputShape0, inputShape1, type, ref_num_nodes, ref_num_subgraphs, targetDevice) = this->GetParam();
    init_input_shapes({{{}, {inputShape0, }}, {{}, {inputShape1, }}});

    auto f = ov::test::snippets::AddSinhFunction({inputShape0, inputShape1});
    function = f.getOriginal();
    setInferenceType(type);
}

std::string AddSinhConst::getTestCaseName(testing::TestParamInfo<ov::test::snippets::AddConstParams> obj) {
    ov::Shape inputShapes, newInputShapes;
    ov::element::Type type;
    std::string targetDevice;
    size_t num_nodes, num_subgraphs;
    std::tie(inputShapes, type, num_nodes, num_subgraphs, targetDevice) = obj.param;

    std::ostringstream result;
    result << "IS[0]=" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "T=" << type << "_";
    result << "#N=" << num_nodes << "_";
    result << "#S=" << num_subgraphs << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void AddSinhConst::SetUp() {
    ov::Shape inputShape;
    ov::element::Type type;
    std::tie(inputShape, type, ref_num_nodes, ref_num_subgraphs, targetDevice) = this->GetParam();
    init_input_shapes({{{}, {inputShape, }}});

    auto f = ov::test::snippets::AddSinhConstFunction({inputShape});
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

TEST_P(AddRollConst, CompareWithRefImpl) {
    run();
    validateNumSubgraphs();
}


} // namespace snippets
} // namespace test
} // namespace ov
