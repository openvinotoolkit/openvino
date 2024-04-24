// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "pass/canonicalization.hpp"
#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "snippets/pass/canonicalization.hpp"
#include "snippets/op/rank_normalization.hpp"
#include <subgraph_simple.hpp>

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
void normalizeResult(const std::shared_ptr<ov::opset1::Result>& res, size_t num_pop) {
    auto rank_norm = std::make_shared<ov::snippets::op::RankNormalization>(res->input_value(0), num_pop);
    res->set_argument(0, rank_norm->output(0));
    res->validate_and_infer_types();
}
} // namespace

CanonicalizationTests::CanonicalizationTests() : TransformationTestsF() {
    // Canonicalization doesn't support tensor names moving on Results since in this pipeline stage it doesn't make sense
    std::shared_ptr<ov::pass::PassConfig> config = std::make_shared<ov::pass::PassConfig>();
    config->disable<ov::pass::CheckUniqueNames>();
    manager = ov::pass::Manager(config);
}

template<typename Func>
void CanonicalizationTests::prepare_functions(const std::vector<VectorDims>& shapes) {
    std::vector<PartialShape> pshapes;
    pshapes.reserve(shapes.size());
    for (const auto& v : shapes )
        pshapes.emplace_back(v);
    const auto &f = Func(pshapes);
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

namespace CanonicalizationTestsInstantiation {

TEST_F(CanonicalizationTests, smoke_Snippets_Canonicalization_AddFunction_0) {
    m_input_shapes = {{2, 3, 10, 64}, {2, 3, 10, 64}};
    m_input_layouts = {{0, 1, 2, 3}, {0, 1, 2, 3}};
    prepare_functions<AddFunction>(m_input_shapes);
    run();
}

TEST_F(CanonicalizationTests, smoke_Snippets_Canonicalization_AddFunction_1) {
    m_input_shapes = {{2,  3, 10, 64},
                      {10, 64}};
    m_input_layouts = {{0, 1, 2, 3},
                       {0, 1}};
    prepare_functions<AddFunction>(m_input_shapes);
    normalizeParameter(model_ref->get_parameters()[1], 2, 0);
    run();
}

TEST_F(CanonicalizationTests, smoke_Snippets_Canonicalization_AddFunction_2) {
    m_input_shapes = {{2, 3,  10, 64, 16},
                      {1, 10, 64}};
    m_input_layouts = {{0, 1, 2, 3, 1},
                       {0, 1, 2}};
    prepare_functions<AddFunction>({{2, 48, 10, 64}, {1, 10, 64}});
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

TEST_F(CanonicalizationTests, smoke_Snippets_Canonicalization_TwoInputsAndOutputsFunction_0) {
    m_input_shapes = {{10, 64}, {2, 3, 10, 64}};
    m_input_layouts = {{0, 1}, {0, 1, 2, 3}};
    prepare_functions<TwoInputsAndOutputsFunction>(m_input_shapes);
    normalizeParameter(model_ref->get_parameters()[0], 2, 0);
    // ShapeInfer to have updated shapes on Results
    model_ref->validate_nodes_and_infer_types();
    normalizeResult(model_ref->get_results()[0], 2);
    run();
}

TEST_F(CanonicalizationTests, smoke_Snippets_Canonicalization_TwoInputsAndOutputsFunction_1) {
    m_input_shapes = {{1, 10, 64}, {2, 3, 10, 64}};
    m_input_layouts = {{0, 2, 1}, {0, 2, 3, 1}};
    prepare_functions<TwoInputsAndOutputsFunction>(m_input_shapes);
    normalizeParameter(model_ref->get_parameters()[0], 1, 0);
    // ShapeInfer to have updated shapes on Results
    model_ref->validate_nodes_and_infer_types();
    normalizeResult(model_ref->get_results()[0], 1);
    run();
}

TEST_F(CanonicalizationTests, smoke_Snippets_Canonicalization_TwoInputsAndOutputsFunction_2) {
    m_input_shapes = {{2, 3,  10, 64, 16},
                      {1, 10, 64}};
    m_input_layouts = {{0, 1, 2, 3, 1},
                       {0, 1, 2}};
    prepare_functions<TwoInputsAndOutputsFunction>({{2, 48, 10, 64}, {1, 10, 64}});
    const auto& params = model_ref->get_parameters();
    // Note: We can't create functions with mismatching input shapes,
    // so we have to set Parameter shapes after the functions were created
    // This reproduces Snippets pipeline well, since blocked shapes are set after the tokenization
    params[0]->set_partial_shape(PartialShape(m_input_shapes[0]));
    model->get_parameters()[0]->set_partial_shape(PartialShape(m_input_shapes[0]));

    normalizeParameter(params[1], 1, 1);
    model_ref->validate_nodes_and_infer_types();
    // When there are blocked shapes, RankNormalization on Results isn't supported for now
    run();
}

} // namespace CanonicalizationTestsInstantiation
}  // namespace snippets
}  // namespace test
}  // namespace ov
