// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "pass/canonicalization.hpp"
#include "common_test_utils/common_utils.hpp"
#include "snippets/pass/canonicalization.hpp"
#include "snippets/op/rank_normalization.hpp"
#include <subgraph_simple.hpp>
#include "openvino/opsets/opset1.hpp"

namespace ov {
namespace test {
namespace snippets {
namespace {
void normalizeParameter(const std::shared_ptr<ov::opset1::Parameter>& par, size_t num_prepend, size_t num_append) {
    auto target_inputs = par->get_output_target_inputs(0);
    auto rank_norm = std::make_shared<ov::snippets::op::RankNormalization>(par,
                                                                           num_prepend,
                                                                           num_append);
    for (auto& t : target_inputs)
        t.replace_source_output(rank_norm);
}
} // namespace

void CanonicalizationTests::prepare_functions(const std::vector<VectorDims>& shapes) {
    std::vector<PartialShape> pshapes;
    pshapes.reserve(shapes.size());
    for (const auto& v : shapes )
        pshapes.emplace_back(v);
    const auto &f = AddFunction(pshapes);
    model = f.getOriginal();
    model_ref = model->clone();
}

void CanonicalizationTests::run() {
    ASSERT_TRUE(model);
    ASSERT_EQ(m_input_shapes.size(), m_input_layouts.size());
    BlockedShapeVector blocked_input_shapes;
    blocked_input_shapes.reserve(m_input_shapes.size());
    for (size_t i = 0; i < m_input_shapes.size(); i++)
        blocked_input_shapes.emplace_back(m_input_shapes[i], m_input_layouts[i]);
    manager.register_pass<ov::snippets::pass::Canonicalization>(blocked_input_shapes);
    disable_rt_info_check();
}

TEST_F(CanonicalizationTests, smoke_Snippets_Canonicalization_0) {
    m_input_shapes = {{2, 3, 10, 64}, {2, 3, 10, 64}};
    m_input_layouts = {{0, 1, 2, 3}, {0, 1, 2, 3}};
    prepare_functions(m_input_shapes);
    run();
}

namespace CanonicalizationTestsInstantiation {
TEST_F(CanonicalizationTests, smoke_Snippets_Canonicalization_1) {
    m_input_shapes = {{2,  3, 10, 64},
                      {10, 64}};
    m_input_layouts = {{0, 1, 2, 3},
                       {0, 1}};
    prepare_functions(m_input_shapes);
    normalizeParameter(model_ref->get_parameters()[1], 2, 0);
    run();
}

TEST_F(CanonicalizationTests, smoke_Snippets_Canonicalization_2) {
    m_input_shapes = {{2, 3,  10, 64, 16},
                      {1, 10, 64}};
    m_input_layouts = {{0, 1, 2, 3, 1},
                       {0, 1, 2}};
    prepare_functions({{2, 48, 10, 64},
                       {1, 10, 64}});
    const auto& params = model_ref->get_parameters();
    // Note: We can't create functions with mismatching input shapes,
    // so we have to set Parameter shapes after the functions were created
    // This reproduces Snippets pipeline well, since blocked shapes are set after the tokenization
    params[0]->set_partial_shape(PartialShape(m_input_shapes[0]));
    model->get_parameters()[0]->set_partial_shape(PartialShape(m_input_shapes[0]));

    normalizeParameter(params[1], 1, 1);
    // need to trigger validate..(...) manually to propagate new blocked shapes,
    // this is correct since RankNormalization ops re-enables shape propagation for blocked shapes
    model_ref->validate_nodes_and_infer_types();
    run();
}

} // namespace CanonicalizationTestsInstantiation
}  // namespace snippets
}  // namespace test
}  // namespace ov
