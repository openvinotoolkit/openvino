// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <unordered_map>

#include <openvino/core/model.hpp>
#include <openvino/pass/manager.hpp>
#include <transformations/rt_info/disable_precision_conversion.hpp>

#include "plugin/transformations/disable_fp16_comp_flux2_rope.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/split.hpp"

using namespace testing;
using namespace ov::intel_gpu;

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;

namespace {

// Runs the pass and asserts the FP16-conversion-disabled state of each named node.
void run_test(const std::shared_ptr<ov::Model>& model,
              const std::unordered_map<std::string, bool>& expected_fp16_disabled_status) {
    ov::pass::Manager manager;
    manager.register_pass<DisableFP16CompFlux2RoPEPattern>();
    manager.run_passes(model);

    for (const auto& op : model->get_ops()) {
        auto it = expected_fp16_disabled_status.find(op->get_friendly_name());
        if (it == expected_fp16_disabled_status.end())
            continue;
        if (it->second) {
            ASSERT_TRUE(ov::is_conversion_disabled(op, ov::element::f16))
                << "FP16 conversion should be disabled for node: " << op->get_friendly_name();
        } else {
            ASSERT_FALSE(ov::is_conversion_disabled(op, ov::element::f16))
                << "FP16 conversion is unexpectedly disabled for node: " << op->get_friendly_name();
        }
    }
}

// Builds a cos/sin table subgraph: Add(param, const) -> Cos|Sin, so the backward
// walk has a non-trivial chain to mark (the Add), with Param/Const skipped.
std::shared_ptr<ov::Node> make_table(bool use_sin, const std::string& prefix) {
    auto param = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{1, 8, 4, 16});
    auto bias = v0::Constant::create(ov::element::f32, ov::Shape{1, 8, 4, 16}, {0.5f});
    auto add = std::make_shared<v1::Add>(param, bias);
    add->set_friendly_name(prefix + "_src");
    std::shared_ptr<ov::Node> table;
    if (use_sin)
        table = std::make_shared<v0::Sin>(add);
    else
        table = std::make_shared<v0::Cos>(add);
    table->set_friendly_name(prefix);
    return table;
}

// Full decomposed FLUX.2 RoPE application: y = x * cos + rotate_half(x) * sin.
std::shared_ptr<ov::Model> create_rope_model(ov::ParameterVector& params_out) {
    auto x = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{1, 8, 4, 16});

    // rotate_half(x)
    auto x1_shape = v0::Constant::create(ov::element::i64, ov::Shape{5}, {1, 8, 4, 8, 2});
    auto x1 = std::make_shared<v1::Reshape>(x, x1_shape, false);
    auto split_axis = v0::Constant::create(ov::element::i64, ov::Shape{}, {-1});
    auto split = std::make_shared<v1::Split>(x1, split_axis, 2);
    auto neg_const = v0::Constant::create(ov::element::f32, ov::Shape{}, {-1.0f});
    auto neg = std::make_shared<v1::Multiply>(split->output(1), neg_const);
    auto x2 = std::make_shared<v0::Concat>(ov::OutputVector{neg, split->output(0)}, -1);
    auto x3_shape = v0::Constant::create(ov::element::i64, ov::Shape{4}, {1, 8, 4, 16});
    auto x3 = std::make_shared<v1::Reshape>(x2, x3_shape, false);

    auto cos_tab = make_table(/*use_sin=*/false, "cos");
    auto sin_tab = make_table(/*use_sin=*/true, "sin");

    auto y1 = std::make_shared<v1::Multiply>(x, cos_tab);
    y1->set_friendly_name("y1");
    auto y2 = std::make_shared<v1::Multiply>(x3, sin_tab);
    y2->set_friendly_name("y2");
    auto result = std::make_shared<v1::Add>(y1, y2);
    result->set_friendly_name("result");

    // Collect every parameter feeding the graph.
    params_out = {x};
    for (const auto& tab : {cos_tab, sin_tab}) {
        auto p = ov::as_type_ptr<v0::Parameter>(tab->get_input_node_shared_ptr(0)->get_input_node_shared_ptr(0));
        if (p)
            params_out.push_back(p);
    }
    return std::make_shared<ov::Model>(ov::OutputVector{result}, params_out);
}

}  // namespace

// The cos/sin tables (and their producer chains) must be kept in FP32, while the
// activation path (x, the multiplies, the add) must not be marked.
TEST(TransformationTests, DisableFP16CompFlux2RoPE_Positive) {
    ov::ParameterVector params;
    auto model = create_rope_model(params);
    run_test(model,
             {{"cos", true},
              {"cos_src", true},
              {"sin", true},
              {"sin_src", true},
              {"y1", false},
              {"y2", false},
              {"result", false}});
}

// A standalone MatMul -> Cos that is not part of a RoPE application must not be marked.
TEST(TransformationTests, DisableFP16CompFlux2RoPE_NegativeMatMulCos) {
    auto lhs = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{16, 1});
    auto rhs = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{1, 64});
    auto matmul = std::make_shared<v0::MatMul>(lhs, rhs);
    matmul->set_friendly_name("matmul");
    auto cos = std::make_shared<v0::Cos>(matmul);
    cos->set_friendly_name("trig");

    auto model = std::make_shared<ov::Model>(ov::OutputVector{cos}, ov::ParameterVector{lhs, rhs});
    run_test(model, {{"trig", false}, {"matmul", false}});
}

// A standalone Concat -> Sin that is not part of a RoPE application must not be marked.
TEST(TransformationTests, DisableFP16CompFlux2RoPE_NegativeConcatSin) {
    auto a = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{16, 32});
    auto b = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{16, 32});
    auto concat = std::make_shared<v0::Concat>(ov::OutputVector{a, b}, -1);
    concat->set_friendly_name("concat");
    auto sin = std::make_shared<v0::Sin>(concat);
    sin->set_friendly_name("trig");

    auto model = std::make_shared<ov::Model>(ov::OutputVector{sin}, ov::ParameterVector{a, b});
    run_test(model, {{"trig", false}, {"concat", false}});
}
