// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/snippets/x64/pass/mark_approximate_softmax_exp.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>

#include "cache/multi_cache.h"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "emitters/snippets/x64/cpu_generator.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/type.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/pass/manager.hpp"
#include "snippets/op/powerstatic.hpp"
#include "snippets/op/reduce.hpp"
#include "snippets/op/result.hpp"
#include "snippets/op/subgraph.hpp"
#include "snippets/pass/manager.hpp"
#include "snippets/pass/positioned_pass.hpp"
#include "snippets/pass/softmax_decomposition.hpp"
#include "transformations/snippets/common/op/fused_mul_add.hpp"
#include "transformations/snippets/common/pass/mul_add_to_fma.hpp"
#include "utils/rt_info/approximate_exp_attribute.hpp"

namespace ov::test::snippets {

namespace {

constexpr size_t softmax_axis = 3;
const ov::PartialShape shape{1, 4, 16, 16};

std::shared_ptr<ov::Model> decomposed_softmax() {
    const auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    const auto softmax = std::make_shared<ov::op::v8::Softmax>(data, static_cast<int64_t>(softmax_axis));
    auto model = std::make_shared<ov::Model>(ov::OutputVector{softmax}, ov::ParameterVector{data});
    ov::pass::Manager manager;
    manager.register_pass<ov::snippets::pass::SoftmaxDecomposition>();
    manager.run_passes(model);
    return model;
}

std::shared_ptr<ov::Node> find_exp(const std::shared_ptr<ov::Model>& model) {
    for (const auto& op : model->get_ordered_ops()) {
        if (ov::is_type<ov::op::v0::Exp>(op)) {
            return op;
        }
    }
    return nullptr;
}

void run_marking(const std::shared_ptr<ov::Model>& model) {
    ov::pass::Manager manager;
    manager.register_pass<ov::intel_cpu::pass::MarkApproximateSoftmaxExp>();
    manager.run_passes(model);
}

// Exp -> ReduceSum -> PowerStatic(power) -> Multiply(Exp, .), with the reciprocal optionally taken
// over a different exponential. The numerator is returned alongside the model: asserting on
// "the first Exp in the graph" instead would not distinguish the two exponentials of the
// sum_of_the_same_exp == false case, and the test would pass with the check it exists to pin
// removed.
struct HandBuilt {
    std::shared_ptr<ov::Model> model;
    std::shared_ptr<ov::Node> numerator;
};

HandBuilt hand_built_softmax(float power, bool extra_consumer) {
    const auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    const auto exp = std::make_shared<ov::op::v0::Exp>(data);
    const auto reduce_sum = std::make_shared<ov::snippets::op::ReduceSum>(exp, softmax_axis);
    const auto reciprocal = std::make_shared<ov::snippets::op::PowerStatic>(reduce_sum, power);
    const auto multiply = std::make_shared<ov::op::v1::Multiply>(exp, reciprocal);
    ov::OutputVector results{multiply};
    if (extra_consumer) {
        results.push_back(exp);
    }
    return {std::make_shared<ov::Model>(results, ov::ParameterVector{data}), exp};
}

// The numerator still feeds its own row sum -- so the consumer count is the same two as a match --
// but the reciprocal that scales it is taken over the sum of a different exponential. Only the
// check that the two sums are the same one can reject this, which is what makes it discriminating.
HandBuilt reciprocal_of_a_foreign_sum() {
    const auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    const auto exp = std::make_shared<ov::op::v0::Exp>(data);
    const auto own_sum = std::make_shared<ov::snippets::op::ReduceSum>(exp, softmax_axis);
    const auto other_exp = std::make_shared<ov::op::v0::Exp>(data);
    const auto foreign_sum = std::make_shared<ov::snippets::op::ReduceSum>(other_exp, softmax_axis);
    const auto reciprocal = std::make_shared<ov::snippets::op::PowerStatic>(foreign_sum, -1.F);
    const auto multiply = std::make_shared<ov::op::v1::Multiply>(exp, reciprocal);
    return {std::make_shared<ov::Model>(ov::OutputVector{multiply, own_sum}, ov::ParameterVector{data}), exp};
}

// A snippets Subgraph whose body is a Softmax whose result is then added to something -- the shape
// the CPU plugin hands to the snippets data-flow pipeline. Built rather than tokenized so the test
// does not depend on the tokenizer's heuristics. The trailing Add is what makes the position of the
// marking pass observable: MulAddToFMA folds the softmax's normalising Multiply into a FusedMulAdd,
// so a pass that runs after it no longer sees the pattern.
std::shared_ptr<ov::snippets::op::Subgraph> softmax_subgraph(const ov::intel_cpu::MultiCachePtr& cache) {
    const auto outer = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    const auto outer_bias = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    const auto body_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    const auto bias = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    const auto softmax = std::make_shared<ov::op::v8::Softmax>(body_param, static_cast<int64_t>(softmax_axis));
    const auto add = std::make_shared<ov::op::v1::Add>(softmax, bias);
    // PropagatePrecision looks the body's output op up in the target machine's jitter table, which
    // only knows the snippets Result, not ov::op::v0::Result.
    const auto result = std::make_shared<ov::snippets::op::Result>(add);
    const auto body = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{body_param, bias});
    auto subgraph = std::make_shared<ov::snippets::op::Subgraph>(ov::OutputVector{outer, outer_bias}, body);
    // PropagatePrecision, which the pipeline registers unconditionally, asks the generator for the
    // target machine, so the Subgraph needs one before any data-flow pass can run. The generator
    // only holds a weak reference to the kernel cache, so the caller has to keep it alive.
    subgraph->set_generator(std::make_shared<ov::intel_cpu::CPUGenerator>(dnnl::impl::cpu::x64::avx2, cache));
    return subgraph;
}

// The two backend passes of Subgraph::getDataFlowPasses that interact, at the positions it gives
// them: MulAddToFMA at the end of the pipeline, and the marking directly after SoftmaxDecomposition.
// MulAddToFMA is listed first so that a marking pass moved to the end of the pipeline lands after
// the fold rather than before it. That is the reverse of the order Subgraph::getDataFlowPasses
// registers them in; the order matters here only under that mutation, so do not "tidy" it to match
// production -- doing so disarms this test silently.
std::vector<ov::snippets::pass::Manager::PositionedPassBase> backend_passes() {
    return {{ov::snippets::pass::PassPosition(ov::snippets::pass::PassPosition::Place::PipelineEnd),
             std::make_shared<ov::intel_cpu::pass::MulAddToFMA>()},
            {ov::snippets::pass::PassPosition(ov::snippets::pass::PassPosition::Place::After,
                                              ov::snippets::pass::SoftmaxDecomposition::get_type_info_static()),
             std::make_shared<ov::intel_cpu::pass::MarkApproximateSoftmaxExp>()}};
}

std::vector<ov::snippets::pass::Manager::PositionedPassBase> without_marking() {
    return {{ov::snippets::pass::PassPosition(ov::snippets::pass::PassPosition::Place::PipelineEnd),
             std::make_shared<ov::intel_cpu::pass::MulAddToFMA>()}};
}

}  // namespace

TEST(MarkApproximateSoftmaxExp, marks_the_exp_of_a_decomposed_softmax) {
    const auto model = decomposed_softmax();
    const auto exp = find_exp(model);
    ASSERT_NE(exp, nullptr);
    ASSERT_FALSE(ov::intel_cpu::is_approximate_exp(exp));

    run_marking(model);

    EXPECT_TRUE(ov::intel_cpu::is_approximate_exp(exp));
}

TEST(MarkApproximateSoftmaxExp, leaves_an_exp_that_is_not_a_softmax_numerator) {
    const auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    const auto exp = std::make_shared<ov::op::v0::Exp>(data);
    const auto model = std::make_shared<ov::Model>(ov::OutputVector{exp}, ov::ParameterVector{data});

    run_marking(model);

    EXPECT_FALSE(ov::intel_cpu::is_approximate_exp(exp));
}

TEST(MarkApproximateSoftmaxExp, leaves_an_exp_divided_by_a_sum_of_another_exp) {
    const auto [model, exp] = reciprocal_of_a_foreign_sum();

    run_marking(model);

    EXPECT_FALSE(ov::intel_cpu::is_approximate_exp(exp));
}

TEST(MarkApproximateSoftmaxExp, leaves_an_exp_scaled_by_a_power_that_is_not_the_reciprocal) {
    const auto [model, exp] = hand_built_softmax(-2.F, false);

    run_marking(model);

    EXPECT_FALSE(ov::intel_cpu::is_approximate_exp(exp));
}

TEST(MarkApproximateSoftmaxExp, leaves_an_exp_that_is_also_read_unnormalised) {
    const auto [model, exp] = hand_built_softmax(-1.F, true);

    run_marking(model);

    EXPECT_FALSE(ov::intel_cpu::is_approximate_exp(exp));
}

TEST(MarkApproximateSoftmaxExp, hand_built_positive_matches_the_decomposed_one) {
    // Pins that the four negatives above differ from a match in exactly the property each names,
    // rather than in the way the graph was built.
    const auto [model, exp] = hand_built_softmax(-1.F, false);

    run_marking(model);

    EXPECT_TRUE(ov::intel_cpu::is_approximate_exp(exp));
}

// The guard the previous version of this feature lacked. Every other test here drives the matcher
// standalone, which is exactly the shape of test that stayed green while the selection rule could
// never fire. This one runs the real snippets data-flow pipeline with the pass at the position the
// plugin gives it, so it fails if the anchor pass is renamed or removed, and -- given the order the
// two backend passes are listed in above -- if the marking is moved to the end of the pipeline,
// where MulAddToFMA has already folded the pattern away.
TEST(MarkApproximateSoftmaxExp, fires_at_its_registered_position_in_the_snippets_pipeline) {
    const auto cache = std::make_shared<ov::intel_cpu::MultiCache>(1);
    const auto subgraph = softmax_subgraph(cache);

    subgraph->data_flow_transformations({}, {}, {}, backend_passes());

    // The fold is what makes the position observable, so a test that stopped provoking it would
    // stop guarding anything.
    const auto ops = subgraph->body_ptr()->get_ordered_ops();
    ASSERT_TRUE(std::any_of(ops.begin(), ops.end(), [](const std::shared_ptr<ov::Node>& op) {
        return ov::is_type<ov::intel_cpu::FusedMulAdd>(op);
    }));
    const auto exp = find_exp(subgraph->body_ptr());
    ASSERT_NE(exp, nullptr);
    EXPECT_TRUE(ov::intel_cpu::is_approximate_exp(exp));
}

// The same pipeline with the pass not registered -- what every caller that did not ask for the
// approximation gets. Pins that the mark comes from the pass and not from the pipeline itself.
TEST(MarkApproximateSoftmaxExp, leaves_the_pipeline_alone_when_it_is_not_registered) {
    const auto cache = std::make_shared<ov::intel_cpu::MultiCache>(1);
    const auto subgraph = softmax_subgraph(cache);

    subgraph->data_flow_transformations({}, {}, {}, without_marking());

    const auto exp = find_exp(subgraph->body_ptr());
    ASSERT_NE(exp, nullptr);
    EXPECT_FALSE(ov::intel_cpu::is_approximate_exp(exp));
}

}  // namespace ov::test::snippets
