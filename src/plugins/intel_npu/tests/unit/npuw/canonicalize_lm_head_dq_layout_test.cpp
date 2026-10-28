// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "npuw_transformations/canonicalize_lm_head_dq_layout.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/manager.hpp"
#include "partitioning/patterns/opt.hpp"

namespace {

constexpr size_t OC = 64;  // stands in for the 151936-wide vocab
constexpr size_t IC = 32;  // stands in for hidden size 2560

using ConstPtr = std::shared_ptr<ov::op::v0::Constant>;

std::shared_ptr<ov::op::v1::Reshape> make_reshape(const ov::Output<ov::Node>& in, const std::vector<int64_t>& shape) {
    return std::make_shared<ov::op::v1::Reshape>(in,
                                                 ov::op::v0::Constant::create(ov::element::i64,
                                                                              ov::Shape{shape.size()},
                                                                              shape),
                                                 true);
}

struct WebNNChain {
    ConstPtr weight, zerop, scale;
    std::shared_ptr<ov::Model> model;
};

// Exactly the chain found in the WebNN-exported Qwen3-4B lm_head submodel:
//   W u8[OC,1,IC] -> Convert ------------------------------+
//   Z u8[OC] -> Convert -> Reshape[OC,1,1] -----------------+-> Subtract
//   S f16[OC] -> Reshape[OC,1,1] ------------------------------> Multiply
//     -> Reshape[OC,IC] -> Transpose[IC,OC] -> MatMul(transpose_b=false) -> Result
WebNNChain build_webnn_chain(const ov::element::Type& weight_type = ov::element::u8) {
    auto weight = std::make_shared<ov::op::v0::Constant>(weight_type, ov::Shape{OC, 1, IC}, 3);
    auto zerop = std::make_shared<ov::op::v0::Constant>(ov::element::u8, ov::Shape{OC}, 1);
    auto scale = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{OC}, 0.5f);

    auto cvt_w = std::make_shared<ov::op::v0::Convert>(weight, ov::element::f16);
    auto cvt_z = std::make_shared<ov::op::v0::Convert>(zerop, ov::element::f16);
    auto zp_rs = make_reshape(cvt_z, {0, 1, 1});
    auto sub = std::make_shared<ov::op::v1::Subtract>(cvt_w, zp_rs);
    auto sc_rs = make_reshape(scale, {0, 1, 1});
    auto mul = std::make_shared<ov::op::v1::Multiply>(sub, sc_rs);
    auto w2d = make_reshape(mul, {static_cast<int64_t>(OC), static_cast<int64_t>(IC)});
    auto perm = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, 0});
    auto transpose = std::make_shared<ov::op::v1::Transpose>(w2d, perm);

    auto act = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 1, IC});
    auto mm = std::make_shared<ov::op::v0::MatMul>(act, transpose, false, false);
    mm->set_friendly_name("logits_matmul");
    mm->output(0).set_names({"logits"});
    auto result = std::make_shared<ov::op::v0::Result>(mm);
    return {weight,
            zerop,
            scale,
            std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{act}, "lm_head")};
}

bool run_pass(const std::shared_ptr<ov::Model>& model) {
    ov::pass::Manager manager;
    manager.register_pass<ov::npuw::CanonicalizeLMHeadDQLayout>();
    return manager.run_passes(model);
}

std::vector<ConstPtr> run_preserve_matcher(const std::shared_ptr<ov::Model>& model) {
    std::vector<ConstPtr> to_keep;
    ov::npuw::patterns::opt::Context ctx(model, model->get_parameters().size());
    ctx.mm_gate = false;
    ov::pass::Manager manager;
    manager.register_pass<ov::npuw::patterns::opt::PreserveConstDictMatMulAsymm>(std::ref(ctx), std::ref(to_keep));
    manager.run_passes(model);
    return to_keep;
}

size_t count_ops(const std::shared_ptr<ov::Model>& model, const ov::DiscreteTypeInfo& type) {
    size_t n = 0;
    for (auto&& op : model->get_ops()) {
        n += op->get_type_info() == type ? 1 : 0;
    }
    return n;
}

}  // namespace

// Reference: the WebNN chain is NOT recognized by the tail-vocab matcher before the pass.
TEST(CanonicalizeLMHeadDQLayout, WebNNChainIsNotMatchedBeforeCanonicalization) {
    auto chain = build_webnn_chain();
    EXPECT_TRUE(run_preserve_matcher(chain.model).empty());
}

TEST(CanonicalizeLMHeadDQLayout, RewritesToCanonicalLayout) {
    auto chain = build_webnn_chain();
    ASSERT_TRUE(run_pass(chain.model));
    chain.model->validate_nodes_and_infer_types();

    EXPECT_EQ(count_ops(chain.model, ov::op::v1::Reshape::get_type_info_static()), 0u);
    EXPECT_EQ(count_ops(chain.model, ov::op::v1::Transpose::get_type_info_static()), 0u);

    auto result = chain.model->get_results().front();
    auto mm = ov::as_type_ptr<ov::op::v0::MatMul>(result->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(mm);
    EXPECT_FALSE(mm->get_transpose_a());
    EXPECT_TRUE(mm->get_transpose_b());
    EXPECT_EQ(mm->get_friendly_name(), "logits_matmul");
    EXPECT_EQ(mm->output(0).get_names(), std::unordered_set<std::string>{"logits"});
    EXPECT_EQ(result->get_output_shape(0), (ov::Shape{1, 1, OC}));
    EXPECT_EQ(mm->get_input_shape(1), (ov::Shape{OC, IC}));
}

TEST(CanonicalizeLMHeadDQLayout, ConstantsStayCompressedAndShareData) {
    auto chain = build_webnn_chain();
    ASSERT_TRUE(run_pass(chain.model));

    std::vector<ConstPtr> consts;
    for (auto&& op : chain.model->get_ordered_ops()) {
        if (auto c = ov::as_type_ptr<ov::op::v0::Constant>(op)) {
            consts.push_back(c);
        }
    }
    ASSERT_EQ(consts.size(), 3u);

    for (auto&& c : consts) {
        if (c->get_element_type() == ov::element::u8 && c->get_shape() == ov::Shape{OC, IC}) {
            EXPECT_EQ(c->get_data_ptr(), chain.weight->get_data_ptr());  // no copy of the weight
        } else if (c->get_element_type() == ov::element::u8) {
            EXPECT_EQ(c->get_shape(), (ov::Shape{OC, 1}));
        } else {
            EXPECT_EQ(c->get_element_type(), ov::element::f16);
            EXPECT_EQ(c->get_shape(), (ov::Shape{OC, 1}));
        }
    }
}

// The whole point: after normalization the existing matcher keeps W / zp / scale as Const.
TEST(CanonicalizeLMHeadDQLayout, CanonicalizedChainIsMatchedByPreserveConstDictMatMulAsymm) {
    auto chain = build_webnn_chain();
    ASSERT_TRUE(run_pass(chain.model));

    auto to_keep = run_preserve_matcher(chain.model);
    ASSERT_EQ(to_keep.size(), 3u);
    for (auto&& c : to_keep) {
        const bool is_weight = c->get_element_type() == ov::element::u8 && c->get_shape().back() == IC;
        EXPECT_EQ(c->get_shape(), is_weight ? ov::Shape({OC, IC}) : ov::Shape({OC, 1}));
    }
}

TEST(CanonicalizeLMHeadDQLayout, IsIdempotentOnCanonicalChain) {
    auto chain = build_webnn_chain();
    ASSERT_TRUE(run_pass(chain.model));
    EXPECT_FALSE(run_pass(chain.model));
}

// Not u8 (e.g. fp16 weight with no real quantization) -> leave the graph alone.
TEST(CanonicalizeLMHeadDQLayout, NonU8WeightIsLeftUntouched) {
    auto chain = build_webnn_chain(ov::element::i8);
    EXPECT_FALSE(run_pass(chain.model));
    EXPECT_EQ(count_ops(chain.model, ov::op::v1::Reshape::get_type_info_static()), 3u);
    EXPECT_EQ(count_ops(chain.model, ov::op::v1::Transpose::get_type_info_static()), 1u);
}

// A plain MatMul(transpose_b=false) on a non-dequantized weight must not be touched.
TEST(CanonicalizeLMHeadDQLayout, PlainMatMulIsLeftUntouched) {
    auto weight = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{IC, OC}, 1.0f);
    auto act = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 1, IC});
    auto mm = std::make_shared<ov::op::v0::MatMul>(act, weight, false, false);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(mm)},
                                             ov::ParameterVector{act});
    EXPECT_FALSE(run_pass(model));
}
