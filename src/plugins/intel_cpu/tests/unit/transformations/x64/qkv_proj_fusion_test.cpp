// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/test_common.hpp"
#include "transformations/cpu_opset/x64/pass/qkv_proj_fusion.hpp"
#include "transformations/cpu_opset/x64/op/qkv_proj.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/visualize_tree.hpp"

using namespace testing;
using namespace ov::pass;
using namespace ov::op;
using namespace ov;

TEST_F(TransformationTestsF, QKVProjFusion1Test) {
    disable_rt_info_check();
    disable_result_friendly_names_check();

    auto is_quantized_int8 = false;
    size_t hidden_size = 2048;
    size_t q_proj_size = 2048;
    size_t k_proj_size = 256;
    size_t v_proj_size = 256;
    auto weights_combined = false;
    {
        auto input_multiply_const = std::make_shared<v0::Constant>(element::f32, Shape{1, 1, hidden_size});
        auto input_param = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, -1, static_cast<int>(hidden_size)});
        auto input_multiply = std::make_shared<v1::Multiply>(input_multiply_const, input_param);

        auto q_proj_weight_const = std::make_shared<v0::Constant>(element::f16, Shape{q_proj_size, hidden_size});
        auto q_proj_weight_cvt = std::make_shared<v0::Convert>(q_proj_weight_const, element::f32);
        auto q_proj = std::make_shared<v0::MatMul>(input_multiply, q_proj_weight_cvt, false, true);

        auto k_proj_weight_const = std::make_shared<v0::Constant>(element::f16, Shape{k_proj_size, hidden_size});
        auto k_proj_weight_cvt = std::make_shared<v0::Convert>(k_proj_weight_const, element::f32);
        auto k_proj = std::make_shared<v0::MatMul>(input_multiply, k_proj_weight_cvt, false, true);

        auto v_proj_weight_const = std::make_shared<v0::Constant>(element::f16, Shape{v_proj_size, hidden_size});
        auto v_proj_weight_cvt = std::make_shared<v0::Convert>(v_proj_weight_const, element::f32);
        auto v_proj = std::make_shared<v0::MatMul>(input_multiply, v_proj_weight_cvt, false, true);

        model = std::make_shared<ov::Model>(OutputVector{q_proj, k_proj, v_proj}, ParameterVector{input_param});
        manager.register_pass<ov::intel_cpu::QKVProjFusion>();
        manager.get_pass_config()->set_callback<ov::intel_cpu::QKVProjFusionPass1>(
            [=](const std::shared_ptr<const ov::Node>) -> bool {
                return true;
            });
    }
    {
        auto input_multiply_const = std::make_shared<v0::Constant>(element::f32, Shape{1, 1, hidden_size});
        auto input_param = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, -1, static_cast<int>(hidden_size)});
        auto input_multiply = std::make_shared<v1::Multiply>(input_multiply_const, input_param);

        auto q_proj_weight_const = std::make_shared<v0::Constant>(element::f16, Shape{q_proj_size, hidden_size});
        auto k_proj_weight_const = std::make_shared<v0::Constant>(element::f16, Shape{k_proj_size, hidden_size});
        auto v_proj_weight_const = std::make_shared<v0::Constant>(element::f16, Shape{v_proj_size, hidden_size});

        intel_cpu::QKVProjectionNode::Config config {is_quantized_int8, static_cast<int>(hidden_size),
                                                                        static_cast<int>(q_proj_size),
                                                                        static_cast<int>(k_proj_size),
                                                                        static_cast<int>(v_proj_size),
                                                                        weights_combined};
        auto qkv_proj = std::make_shared<intel_cpu::QKVProjectionNode>(OutputVector{input_multiply,
                                                                                    q_proj_weight_const,
                                                                                    k_proj_weight_const,
                                                                                    v_proj_weight_const},
                                                                                    config);

        auto q_proj = std::make_shared<v0::Result>(qkv_proj->output(0));
        auto k_proj = std::make_shared<v0::Result>(qkv_proj->output(1));
        auto v_proj = std::make_shared<v0::Result>(qkv_proj->output(2));
        model_ref = std::make_shared<ov::Model>(OutputVector{q_proj, k_proj, v_proj}, ParameterVector{input_param});
    }
}

namespace {

constexpr size_t hidden_size = 2048;

std::shared_ptr<v0::Constant> make_weight(element::Type type, const Shape& shape) {
    return v0::Constant::create(type, shape, std::vector<float>(shape_size(shape), 0.01f));
}

// Per-output-channel int8 weight: Multiply(Convert(i8 [rows, hidden]), f32 scales [scale_rows, 1]).
std::shared_ptr<Node> make_int8_weight(size_t rows,
                                       size_t scale_rows,
                                       std::shared_ptr<v0::Constant>& w,
                                       std::shared_ptr<v0::Constant>& scales) {
    w = v0::Constant::create(element::i8, Shape{rows, hidden_size}, std::vector<int8_t>(rows * hidden_size, 1));
    scales = v0::Constant::create(element::f32, Shape{scale_rows, 1}, std::vector<float>(scale_rows, 0.01f));
    return std::make_shared<v1::Multiply>(std::make_shared<v0::Convert>(w, element::f32), scales);
}

std::shared_ptr<Model> make_fused_ref(const std::shared_ptr<v0::Parameter>& input,
                                      const OutputVector& weights,
                                      const OutputVector& scales,
                                      const std::vector<size_t>& proj_sizes,
                                      bool weights_combined) {
    const intel_cpu::QKVProjectionNode::Config config{!scales.empty(),
                                                      static_cast<int>(hidden_size),
                                                      static_cast<int>(proj_sizes[0]),
                                                      static_cast<int>(proj_sizes[1]),
                                                      static_cast<int>(proj_sizes[2]),
                                                      weights_combined};
    OutputVector args{input};
    args.insert(args.end(), weights.begin(), weights.end());
    args.insert(args.end(), scales.begin(), scales.end());
    auto qkv = std::make_shared<intel_cpu::QKVProjectionNode>(args, config);
    return std::make_shared<Model>(qkv->outputs(), ParameterVector{input});
}

template <class TPass>
void register_qkv_fusion(pass::Manager& manager, bool supported = true) {
    manager.register_pass<intel_cpu::QKVProjFusion>();
    manager.get_pass_config()->set_callback<TPass>([supported](const std::shared_ptr<const Node>&) {
        return supported;
    });
}

// ---------------------------------------------------------------------------------------------------------------
// QKVProjectionNode shape inference

TEST(QKVProjectionNodeTest, ShapeInferenceRank2And3) {
    const intel_cpu::QKVProjectionNode::Config config{false, static_cast<int>(hidden_size), 2048, 256, 256, true};
    for (const auto& shape : {PartialShape{-1, static_cast<int64_t>(hidden_size)},
                              PartialShape{-1, -1, static_cast<int64_t>(hidden_size)},
                              PartialShape{7, static_cast<int64_t>(hidden_size)}}) {
        auto input = std::make_shared<v0::Parameter>(element::f32, shape);
        auto w = make_weight(element::f16, Shape{2560, hidden_size});
        auto qkv = std::make_shared<intel_cpu::QKVProjectionNode>(OutputVector{input, w, w, w}, config);
        ASSERT_EQ(qkv->get_output_size(), 3);
        for (size_t i = 0; i < 3; i++) {
            auto expected = shape;
            expected[shape.size() - 1] = std::vector<int64_t>{2048, 256, 256}[i];
            EXPECT_EQ(qkv->get_output_partial_shape(i), expected) << "output " << i << " for input " << shape;
            EXPECT_EQ(qkv->get_output_element_type(i), element::f32);
        }
    }
}

TEST(QKVProjectionNodeTest, UnsupportedRankThrows) {
    const intel_cpu::QKVProjectionNode::Config config{false, static_cast<int>(hidden_size), 2048, 256, 256, true};
    for (const auto& shape : {PartialShape{static_cast<int64_t>(hidden_size)},
                              PartialShape{1, -1, -1, static_cast<int64_t>(hidden_size)},
                              PartialShape::dynamic()}) {
        auto input = std::make_shared<v0::Parameter>(element::f32, shape);
        auto w = make_weight(element::f16, Shape{2560, hidden_size});
        EXPECT_THROW(std::make_shared<intel_cpu::QKVProjectionNode>(OutputVector{input, w, w, w}, config),
                     ov::NodeValidationFailure)
            << "input " << shape;
    }
}

// ---------------------------------------------------------------------------------------------------------------
// QKVProjFusionPass1: three MatMuls on the same input

struct QKVSeparateParams {
    PartialShape input_shape;
    bool quantized;
    std::vector<size_t> proj_sizes;
};

class QKVProjFusionSeparateTest : public TransformationTestsF, public WithParamInterface<QKVSeparateParams> {};

TEST_P(QKVProjFusionSeparateTest, Fused) {
    disable_rt_info_check();
    disable_result_friendly_names_check();
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    const auto& p = GetParam();
    auto input = std::make_shared<v0::Parameter>(element::f32, p.input_shape);
    OutputVector weights, scales, outputs;
    for (auto rows : p.proj_sizes) {
        std::shared_ptr<Node> w_f32;
        if (p.quantized) {
            std::shared_ptr<v0::Constant> w, s;
            w_f32 = make_int8_weight(rows, rows, w, s);
            weights.push_back(w);
            scales.push_back(s);
        } else {
            auto w = make_weight(element::f16, Shape{rows, hidden_size});
            w_f32 = std::make_shared<v0::Convert>(w, element::f32);
            weights.push_back(w);
        }
        outputs.push_back(std::make_shared<v0::MatMul>(input, w_f32, false, true));
    }
    model = std::make_shared<Model>(outputs, ParameterVector{input});
    register_qkv_fusion<intel_cpu::QKVProjFusionPass1>(manager);
    model_ref = make_fused_ref(input, weights, scales, p.proj_sizes, false);
}

INSTANTIATE_TEST_SUITE_P(smoke,
                         QKVProjFusionSeparateTest,
                         Values(QKVSeparateParams{PartialShape{-1, -1, hidden_size}, false, {2048, 256, 256}},
                                // flattened [tokens, hidden] input
                                QKVSeparateParams{PartialShape{-1, hidden_size}, false, {2048, 256, 256}},
                                QKVSeparateParams{PartialShape{-1, hidden_size}, true, {2048, 2048, 2048}},
                                QKVSeparateParams{PartialShape{-1, -1, hidden_size}, true, {2048, 256, 256}}));

// A per-tensor scale has a single value: the executor would read proj_size values from it.
TEST_F(TransformationTestsF, QKVProjFusionSeparatePerTensorScaleNotFused) {
    auto input = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, hidden_size});
    OutputVector outputs;
    for (auto [rows, scale_rows] : {std::pair<size_t, size_t>{2048, 2048}, {256, 1}, {256, 256}}) {
        std::shared_ptr<v0::Constant> w, s;
        outputs.push_back(std::make_shared<v0::MatMul>(input, make_int8_weight(rows, scale_rows, w, s), false, true));
    }
    model = std::make_shared<Model>(outputs, ParameterVector{input});
    register_qkv_fusion<intel_cpu::QKVProjFusionPass1>(manager);
}

// ---------------------------------------------------------------------------------------------------------------
// QKVProjFusionPass2: one combined q/k/v weight followed by a VariadicSplit on the last axis

enum class WeightForm { F16_CONVERT, BF16_CONVERT, F32_PLAIN, INT8_PER_OC };

struct QKVCombinedParams {
    PartialShape input_shape;
    WeightForm weight;
    std::shared_ptr<v0::Constant> axis;
    element::Type lengths_type;
    std::vector<int64_t> split_lengths;  // as written in the graph, may contain -1
    std::vector<size_t> proj_sizes;      // the resolved q/k/v sizes
};

struct CombinedModel {
    std::shared_ptr<Model> model;
    std::shared_ptr<v0::Parameter> input;
    std::shared_ptr<v0::Constant> weight;
    std::shared_ptr<v0::Constant> scales;
};

CombinedModel make_combined_qkv(const PartialShape& input_shape,
                                WeightForm form,
                                const std::shared_ptr<v0::Constant>& axis,
                                element::Type lengths_type,
                                const std::vector<int64_t>& split_lengths,
                                size_t rows,
                                size_t scale_rows = 0) {
    CombinedModel r;
    r.input = std::make_shared<v0::Parameter>(element::f32, input_shape);
    std::shared_ptr<Node> w_f32;
    switch (form) {
    case WeightForm::F16_CONVERT:
    case WeightForm::BF16_CONVERT:
        r.weight =
            make_weight(form == WeightForm::F16_CONVERT ? element::f16 : element::bf16, Shape{rows, hidden_size});
        w_f32 = std::make_shared<v0::Convert>(r.weight, element::f32);
        break;
    case WeightForm::F32_PLAIN:
        r.weight = make_weight(element::f32, Shape{rows, hidden_size});
        w_f32 = r.weight;
        break;
    case WeightForm::INT8_PER_OC:
        w_f32 = make_int8_weight(rows, scale_rows ? scale_rows : rows, r.weight, r.scales);
        break;
    }
    auto qkv = std::make_shared<v0::MatMul>(r.input, w_f32, false, true);
    auto split = std::make_shared<v1::VariadicSplit>(
        qkv,
        axis,
        v0::Constant::create(lengths_type, Shape{split_lengths.size()}, split_lengths));
    r.model = std::make_shared<Model>(split->outputs(), ParameterVector{r.input});
    return r;
}

std::shared_ptr<v0::Constant> axis_const(element::Type type, int64_t axis, bool as_1d = false) {
    return v0::Constant::create(type, as_1d ? Shape{1} : Shape{}, {axis});
}

class QKVProjFusionCombinedTest : public TransformationTestsF, public WithParamInterface<QKVCombinedParams> {};

TEST_P(QKVProjFusionCombinedTest, Fused) {
    disable_rt_info_check();
    disable_result_friendly_names_check();
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    const auto& p = GetParam();
    const auto rows = p.proj_sizes[0] + p.proj_sizes[1] + p.proj_sizes[2];
    auto m = make_combined_qkv(p.input_shape, p.weight, p.axis, p.lengths_type, p.split_lengths, rows);
    model = m.model;
    register_qkv_fusion<intel_cpu::QKVProjFusionPass2>(manager);
    const OutputVector scales = m.scales ? OutputVector{m.scales, m.scales, m.scales} : OutputVector{};
    model_ref = make_fused_ref(m.input, {m.weight, m.weight, m.weight}, scales, p.proj_sizes, true);
}

const std::vector<size_t> gqa{2048, 256, 256};
const std::vector<int64_t> gqa_lengths{2048, 256, 256};

INSTANTIATE_TEST_SUITE_P(smoke,
                         QKVProjFusionCombinedTest,
                         Values(
                             // vLLM form: flattened [tokens, hidden], bf16 weight, positive axis, i64 lengths, GQA
                             QKVCombinedParams{PartialShape{-1, hidden_size},
                                               WeightForm::BF16_CONVERT,
                                               axis_const(element::i64, 1),
                                               element::i64,
                                               gqa_lengths,
                                               gqa},
                             QKVCombinedParams{PartialShape{-1, hidden_size},
                                               WeightForm::F16_CONVERT,
                                               axis_const(element::i64, -1),
                                               element::i32,
                                               gqa_lengths,
                                               gqa},
                             // rank-3 input, negative and positive last axis
                             QKVCombinedParams{PartialShape{-1, -1, hidden_size},
                                               WeightForm::F16_CONVERT,
                                               axis_const(element::i64, -1),
                                               element::i32,
                                               {2048, 2048, 2048},
                                               {2048, 2048, 2048}},
                             QKVCombinedParams{PartialShape{-1, -1, hidden_size},
                                               WeightForm::BF16_CONVERT,
                                               axis_const(element::i32, 2),
                                               element::i64,
                                               gqa_lengths,
                                               gqa},
                             // axis given as a 1-element tensor
                             QKVCombinedParams{PartialShape{-1, hidden_size},
                                               WeightForm::BF16_CONVERT,
                                               axis_const(element::i32, 1, true),
                                               element::i64,
                                               gqa_lengths,
                                               gqa},
                             // inferred (-1) split length
                             QKVCombinedParams{PartialShape{-1, hidden_size},
                                               WeightForm::BF16_CONVERT,
                                               axis_const(element::i64, 1),
                                               element::i64,
                                               {2048, -1, 256},
                                               gqa},
                             // weight without a Convert
                             QKVCombinedParams{PartialShape{-1, hidden_size},
                                               WeightForm::F32_PLAIN,
                                               axis_const(element::i64, 1),
                                               element::i64,
                                               gqa_lengths,
                                               gqa},
                             // int8 weight with per-output-channel scales
                             QKVCombinedParams{PartialShape{-1, hidden_size},
                                               WeightForm::INT8_PER_OC,
                                               axis_const(element::i64, 1),
                                               element::i64,
                                               gqa_lengths,
                                               gqa},
                             QKVCombinedParams{PartialShape{-1, -1, hidden_size},
                                               WeightForm::INT8_PER_OC,
                                               axis_const(element::i64, -1),
                                               element::i32,
                                               gqa_lengths,
                                               gqa}));

// The split cuts the token axis instead of the output channels.
TEST_F(TransformationTestsF, QKVProjFusionCombinedNotLastAxisNotFused) {
    model = make_combined_qkv(PartialShape{3, hidden_size},
                              WeightForm::BF16_CONVERT,
                              axis_const(element::i64, 0),
                              element::i64,
                              {1, 1, 1},
                              2560)
                .model;
    register_qkv_fusion<intel_cpu::QKVProjFusionPass2>(manager);
}

TEST_F(TransformationTestsF, QKVProjFusionCombinedRank3TokenAxisNotFused) {
    model = make_combined_qkv(PartialShape{-1, 3, hidden_size},
                              WeightForm::F16_CONVERT,
                              axis_const(element::i64, -2),
                              element::i64,
                              {1, 1, 1},
                              2560)
                .model;
    register_qkv_fusion<intel_cpu::QKVProjFusionPass2>(manager);
}

// Two outputs are not a q/k/v split.
TEST_F(TransformationTestsF, QKVProjFusionCombinedTwoWaySplitNotFused) {
    model = make_combined_qkv(PartialShape{-1, hidden_size},
                              WeightForm::BF16_CONVERT,
                              axis_const(element::i64, 1),
                              element::i64,
                              {2048, 512},
                              2560)
                .model;
    register_qkv_fusion<intel_cpu::QKVProjFusionPass2>(manager);
}

// A per-tensor scale has a single value: the executor would read q + k + v values from it.
TEST_F(TransformationTestsF, QKVProjFusionCombinedPerTensorScaleNotFused) {
    model = make_combined_qkv(PartialShape{-1, hidden_size},
                              WeightForm::INT8_PER_OC,
                              axis_const(element::i64, 1),
                              element::i64,
                              gqa_lengths,
                              2560,
                              1)
                .model;
    register_qkv_fusion<intel_cpu::QKVProjFusionPass2>(manager);
}

// The plugin callback (QKVProjection::isSupportedOperation) has the last word.
TEST_F(TransformationTestsF, QKVProjFusionCombinedRejectedByCallbackNotFused) {
    model = make_combined_qkv(PartialShape{-1, hidden_size},
                              WeightForm::BF16_CONVERT,
                              axis_const(element::i64, 1),
                              element::i64,
                              gqa_lengths,
                              2560)
                .model;
    register_qkv_fusion<intel_cpu::QKVProjFusionPass2>(manager, false);
}

}  // namespace

namespace {

// The fused node keeps the runtime info of every node it replaces (q/k/v MatMuls, or the MatMul and the split).
void check_rt_info_propagated(const std::shared_ptr<Model>& model, const NodeVector& fused) {
    for (size_t i = 0; i < fused.size(); i++) {
        fused[i]->get_rt_info()["qkv_test_key_" + std::to_string(i)] = true;
    }
    pass::Manager manager;
    register_qkv_fusion<intel_cpu::QKVProjFusionPass1>(manager);
    manager.get_pass_config()->set_callback<intel_cpu::QKVProjFusionPass2>([](const std::shared_ptr<const Node>&) {
        return true;
    });
    manager.run_passes(model);
    std::shared_ptr<Node> qkv;
    for (const auto& op : model->get_ops()) {
        if (ov::is_type<intel_cpu::QKVProjectionNode>(op)) {
            qkv = op;
        }
    }
    ASSERT_NE(qkv, nullptr);
    for (size_t i = 0; i < fused.size(); i++) {
        EXPECT_EQ(qkv->get_rt_info().count("qkv_test_key_" + std::to_string(i)), 1) << "from node " << i;
    }
}

TEST(QKVProjFusionRtInfoTest, SeparateMatMuls) {
    auto input = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, hidden_size});
    NodeVector matmuls;
    for (size_t rows : {2048, 256, 256}) {
        auto w = std::make_shared<v0::Convert>(make_weight(element::f16, Shape{rows, hidden_size}), element::f32);
        matmuls.push_back(std::make_shared<v0::MatMul>(input, w, false, true));
    }
    auto model = std::make_shared<Model>(OutputVector{matmuls[0], matmuls[1], matmuls[2]}, ParameterVector{input});
    check_rt_info_propagated(model, matmuls);
}

TEST(QKVProjFusionRtInfoTest, CombinedWeight) {
    auto m = make_combined_qkv(PartialShape{-1, hidden_size},
                               WeightForm::BF16_CONVERT,
                               axis_const(element::i64, 1),
                               element::i64,
                               gqa_lengths,
                               2560);
    auto split = m.model->get_results()[0]->get_input_node_shared_ptr(0);
    check_rt_info_propagated(m.model, {split->get_input_node_shared_ptr(0), split});
}

}  // namespace
