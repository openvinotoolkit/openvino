// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <map>
#include <string>
#include <memory>
#include <unordered_set>
#include <utility>
#include <vector>

#include "intel_gpu/op/gemm.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"

#include <openvino/pass/manager.hpp>
#include <openvino/pass/constant_folding.hpp>
#include <openvino/core/model.hpp>
#include "openvino/core/coordinate_diff.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/gather_nd.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/gelu.hpp"
#include "openvino/op/matmul.hpp"
#include "ov_ops/rms.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/scatter_nd_update.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/runtime/tensor.hpp"

#include <plugin/transformations/increase_position_ids_precision.hpp>
#include <plugin/transformations/increase_rms_input_precision.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/convert_precision.hpp>
#include <transformations/fp16_compression/mark_decompression_convert_constant_folding.hpp>
#include <transformations/utils/utils.hpp>
#include "openvino/pass/visualize_tree.hpp"

#include "common_test_utils/ov_test_utils.hpp"

// Sin/Cos discovery helpers and regression tests.
namespace {

void run_discovery(const std::shared_ptr<ov::Model>& model, bool disabled = false) {
    ov::pass::Manager manager;
    manager.register_pass<ov::intel_gpu::IncreasePositionIdsPrecisionForLtxVideo>();
    if (disabled) {
        manager.get_pass_config()->set_callback<ov::intel_gpu::IncreasePositionIdsPrecisionForLtxVideo>(
            [](const std::shared_ptr<const ov::Node>&) { return true; });
    }
    manager.run_passes(model);
}

void expect_fp32_pair(const std::shared_ptr<ov::Model>& model) {
    ASSERT_NO_THROW(model->validate_nodes_and_infer_types());
    ov::OutputVector angles;
    for (size_t i = 0; i < 2; ++i) {
        auto boundary = model->get_results()[i]->input_value(0).get_node_shared_ptr();
        ASSERT_TRUE(ov::is_type<ov::op::v0::Convert>(boundary));
        EXPECT_EQ(boundary->get_output_element_type(0), ov::element::f16);
        const auto trig = boundary->input_value(0).get_node_shared_ptr();
        if (i == 0) {
            ASSERT_TRUE(ov::is_type<ov::op::v0::Sin>(trig));
        } else {
            ASSERT_TRUE(ov::is_type<ov::op::v0::Cos>(trig));
        }
        EXPECT_EQ(trig->get_output_element_type(0), ov::element::f32);
        angles.push_back(trig->input_value(0));
    }
    ASSERT_EQ(angles[0], angles[1]);
    const auto integer_inputs = [](const ov::Output<ov::Node>& root) {
        std::map<ov::Output<ov::Node>, size_t> inputs;
        std::unordered_set<const ov::Node*> visited;
        ov::OutputVector pending{root};
        while (!pending.empty()) {
            const auto node = pending.back().get_node_shared_ptr();
            pending.pop_back();
            if (!visited.insert(node.get()).second) {
                continue;
            }
            for (const auto& input : node->input_values()) {
                if (input.get_element_type().is_real()) {
                    pending.push_back(input);
                } else {
                    ++inputs[input];
                }
            }
        }
        return inputs;
    };
    EXPECT_EQ(integer_inputs(angles[0]), integer_inputs(model->get_results()[2]->input_value(0)));
    std::unordered_set<const ov::Node*> visited;
    while (!angles.empty()) {
        const auto output = angles.back();
        angles.pop_back();
        EXPECT_EQ(output.get_element_type(), ov::element::f32) << output;
        if (!visited.insert(output.get_node()).second) {
            continue;
        }
        for (const auto& input : output.get_node()->input_values()) {
            if (input.get_element_type().is_real()) {
                if (input.get_element_type() == ov::element::f16 &&
                    ov::is_type<ov::op::v0::Parameter>(input.get_node())) {
                    EXPECT_TRUE(ov::is_type<ov::op::v0::Convert>(output.get_node()));
                    continue;
                }
                angles.push_back(input);
            }
        }
    }
    const auto nodes = model->get_ordered_ops();
    run_discovery(model);
    EXPECT_EQ(nodes, model->get_ordered_ops());
}

std::shared_ptr<ov::Model> make_pair(const ov::Output<ov::Node>& angle,
                                    const ov::ParameterVector& parameters) {
    auto sin = std::make_shared<ov::op::v0::Sin>(angle);
    auto cos = std::make_shared<ov::op::v0::Cos>(angle);
    return std::make_shared<ov::Model>(ov::OutputVector{sin, cos, angle}, parameters);
}

}  // namespace

TEST(IncreasePositionIdsLTXVideoDiscoveryTest, TimeEmbeddingAndLayoutVariantsWithoutRoPE) {
    for (const bool layout : {false, true}) {
        SCOPED_TRACE(layout);
        auto coordinates = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{2});
        auto positions = std::make_shared<ov::op::v0::Convert>(coordinates, ov::element::f16);
        auto coefficient = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{2}, {31415.927734f});
        auto compressed = std::make_shared<ov::op::v0::Convert>(coefficient, ov::element::f16);
        auto multiply = std::make_shared<ov::op::v1::Multiply>(positions, compressed);
        ov::Output<ov::Node> angle = multiply;
        if (layout) {
            auto shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, 2});
            angle = std::make_shared<ov::op::v1::Reshape>(angle, shape, false);
        }
        auto model = make_pair(angle, {coordinates});
        auto table = std::make_shared<ov::op::v0::Concat>(
            ov::OutputVector{model->get_results()[0]->input_value(0), model->get_results()[1]->input_value(0)},
            -1);
        model->add_results({std::make_shared<ov::op::v0::Result>(table)});
        run_discovery(model);
        expect_fp32_pair(model);
        EXPECT_EQ(model->get_results()[2]->input_value(0), angle);
        EXPECT_EQ(angle.get_element_type(), ov::element::f16);
        EXPECT_EQ(table->get_output_element_type(0), ov::element::f16);
    }
}

TEST(IncreasePositionIdsLTXVideoDiscoveryTest, IneligiblePairsAndFP32PathsStayUnchanged) {
    for (const std::string mode : {"different", "different_port", "missing", "ambiguous", "ambiguous_sin",
                                   "no_arithmetic", "fp32", "fp32_convert", "disabled", "cos_disabled", "bf16",
                                   "fixed_gemm"}) {
        SCOPED_TRACE(mode);
        auto type = mode == "fp32" || mode == "fp32_convert" ? ov::element::f32 : ov::element::f16;
        auto input = std::make_shared<ov::op::v0::Parameter>(type, ov::Shape{2});
        auto constant = ov::op::v0::Constant::create(type, ov::Shape{2}, {2});
        ov::Output<ov::Node> angle = std::make_shared<ov::op::v1::Add>(input, constant);
        if (mode == "no_arithmetic") {
            angle = input;
        } else if (mode == "fp32_convert") {
            angle = std::make_shared<ov::op::v0::Convert>(angle, ov::element::f32);
        } else if (mode == "bf16") {
            angle = std::make_shared<ov::op::v0::Convert>(angle, ov::element::bf16);
            angle = std::make_shared<ov::op::v0::Convert>(angle, ov::element::f16);
        } else if (mode == "fixed_gemm") {
            angle = std::make_shared<ov::intel_gpu::op::Gemm>(
                input, constant, std::vector<int64_t>{0}, std::vector<int64_t>{0}, std::vector<int64_t>{},
                ov::element::f16);
        }
        auto model = make_pair(angle, {input});
        if (mode == "different") {
            model->get_results()[1]->input_value(0).get_node()->input(0).replace_source_output(input);
        } else if (mode == "ambiguous") {
            model->add_results({std::make_shared<ov::op::v0::Result>(std::make_shared<ov::op::v0::Cos>(angle))});
        } else if (mode == "ambiguous_sin") {
            model->add_results({std::make_shared<ov::op::v0::Result>(std::make_shared<ov::op::v0::Sin>(angle))});
        } else if (mode == "missing") {
            model = std::make_shared<ov::Model>(
                ov::OutputVector{std::make_shared<ov::op::v0::Sin>(angle)}, ov::ParameterVector{input});
        } else if (mode == "different_port") {
            auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
            auto split = std::make_shared<ov::op::v1::Split>(angle, axis, 2);
            model->get_results()[0]->input_value(0).get_node()->input(0).replace_source_output(split->output(0));
            model->get_results()[1]->input_value(0).get_node()->input(0).replace_source_output(split->output(1));
        }
        const auto nodes = model->get_ordered_ops();
        if (mode == "cos_disabled") {
            ov::pass::Manager manager;
            manager.register_pass<ov::intel_gpu::IncreasePositionIdsPrecisionForLtxVideo>();
            manager.get_pass_config()->set_callback<ov::intel_gpu::IncreasePositionIdsPrecisionForLtxVideo>(
                [](const std::shared_ptr<const ov::Node>& node) { return ov::is_type<ov::op::v0::Cos>(node); });
            manager.run_passes(model);
        } else {
            run_discovery(model, mode == "disabled");
        }
        EXPECT_EQ(nodes, model->get_ordered_ops());
        EXPECT_NO_THROW(model->validate_nodes_and_infer_types());
    }
}

// Cross-model synthetic patterns; these do not certify the corresponding real models.
TEST(IncreasePositionIdsLTXVideoCrossModelPatternTest, ReconstructsPreviouslyAnalyzedAngleStructures) {
    for (const std::string pattern : {"generic_rope", "qwen25", "qwen3", "gpt_oss", "direct", "gemm"}) {
        SCOPED_TRACE(pattern);
        auto positions = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{2, 2});
        auto converted = std::make_shared<ov::op::v0::Convert>(positions, ov::element::f16);
        auto frequency = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{2, 2}, {1});
        auto compressed = std::make_shared<ov::op::v0::Convert>(frequency, ov::element::f16);
        ov::Output<ov::Node> angle;
        if (pattern == "gemm") {
            angle = std::make_shared<ov::intel_gpu::op::Gemm>(converted, compressed,
                                                           std::vector<int64_t>{0, 1},
                                                           std::vector<int64_t>{0, 1},
                                                           std::vector<int64_t>{0, 1});
        } else {
            angle = std::make_shared<ov::op::v0::MatMul>(converted, compressed);
        }
        auto order = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, 0});
        if (pattern == "qwen25" || pattern == "gpt_oss") {
            angle = std::make_shared<ov::op::v1::Transpose>(angle, order);
        }
        if (pattern == "qwen3") {
            auto shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {2, 2});
            angle = std::make_shared<ov::op::v1::Reshape>(angle, shape, false);
            auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
            auto indices = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {0, 1});
            auto gather = std::make_shared<ov::op::v8::Gather>(angle, indices, axis);
            auto scatter_indices = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2, 1}, {0, 1});
            angle = std::make_shared<ov::op::v3::ScatterNDUpdate>(angle, scatter_indices, gather);
        }
        if (pattern == "generic_rope" || pattern == "qwen25" || pattern == "qwen3") {
            angle = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{angle, angle}, -1);
        }
        auto model = make_pair(angle, {positions});
        if (pattern == "qwen25" || pattern == "qwen3") {
            auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {-1});
            auto indices = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {0, 1});
            for (size_t i = 0; i < 2; ++i) {
                auto table = std::make_shared<ov::op::v8::Gather>(
                    model->get_results()[i]->input_value(0), indices, axis);
                model->add_results({std::make_shared<ov::op::v0::Result>(table)});
            }
        }
        if (pattern == "gpt_oss") {
            auto scale = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{}, {0.5f});
            for (size_t i = 0; i < 2; ++i) {
                auto result = model->get_results()[i];
                auto scaled = std::make_shared<ov::op::v1::Multiply>(result->input_value(0), scale);
                model->add_results({std::make_shared<ov::op::v0::Result>(scaled)});
            }
        }
        run_discovery(model);
        expect_fp32_pair(model);
        EXPECT_EQ(model->get_results()[2]->input_value(0), angle);
        if (pattern == "gpt_oss") {
            EXPECT_EQ(model->get_results()[3]->get_input_element_type(0), ov::element::f16);
            EXPECT_EQ(model->get_results()[4]->get_input_element_type(0), ov::element::f16);
        }
    }
}

// Discovery regressions through the enclosing pass and FP16 input boundaries.
TEST(IncreasePositionIdsLTXVideoDiscoveryTest, EnclosingPassPreservesTimeEmbeddingAndIsIdempotent) {
    for (const auto type : {ov::element::f16, ov::element::f32}) {
        auto positions = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{2});
        auto converted = std::make_shared<ov::op::v0::Convert>(positions, type);
        auto frequency = ov::op::v0::Constant::create(type, ov::Shape{2}, {2});
        auto angle = std::make_shared<ov::op::v1::Multiply>(converted, frequency);
        auto model = make_pair(angle, {positions});
        const auto original = model->get_ordered_ops();
        ov::pass::Manager manager;
        manager.register_pass<ov::intel_gpu::IncreasePositionIdsPrecision>();
        manager.run_passes(model);
        if (type == ov::element::f16) {
            expect_fp32_pair(model);
        } else {
            EXPECT_EQ(original, model->get_ordered_ops());
        }
        const auto nodes = model->get_ordered_ops();
        manager.run_passes(model);
        EXPECT_EQ(nodes, model->get_ordered_ops());
    }
}

TEST(IncreasePositionIdsLTXVideoDiscoveryTest, ReusesFP16ParameterBoundaryOnRepeatedApplication) {
    for (const bool enclosing : {false, true}) {
        SCOPED_TRACE(enclosing);
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{2});
        auto coefficient = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{2}, {2});
        auto angle = std::make_shared<ov::op::v1::Add>(input, coefficient);
        auto model = make_pair(angle, {input});
        ov::pass::Manager manager;
        if (enclosing) {
            manager.register_pass<ov::intel_gpu::IncreasePositionIdsPrecision>();
        } else {
            manager.register_pass<ov::intel_gpu::IncreasePositionIdsPrecisionForLtxVideo>();
        }
        manager.run_passes(model);
        expect_fp32_pair(model);
        const auto nodes = model->get_ordered_ops();
        manager.run_passes(model);
        EXPECT_EQ(nodes, model->get_ordered_ops());
        EXPECT_EQ(input->get_output_element_type(0), ov::element::f16);
        EXPECT_EQ(model->get_results()[2]->input_value(0), angle->output(0));
    }
}

using namespace testing;
using namespace ov::intel_gpu;

TEST_F(TransformationTestsF, IncreasePositionIdsPrecisionWithoutUnsqueeze) {
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{ -1, -1 });
        auto input_convert = std::make_shared<ov::op::v0::Convert>(input, ov::element::i32);
        auto input_unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(input_convert, std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1}));
        auto input_convert_fp = std::make_shared<ov::op::v0::Convert>(input_unsqueeze, ov::element::f16);
        auto rotary_embd_const = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1, 64, 1});

        auto gemm = std::make_shared<ov::intel_gpu::op::Gemm>(input_convert_fp, rotary_embd_const, std::vector<int64_t>{}, std::vector<int64_t>{}, std::vector<int64_t>{});
        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{gemm, gemm}, 2);

        auto cos = std::make_shared<ov::op::v0::Cos>(concat);
        auto sin = std::make_shared<ov::op::v0::Sin>(concat);

        auto rope_input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape::dynamic(4));
        auto rope = std::make_shared<ov::op::internal::RoPE>(ov::OutputVector{rope_input, cos, sin}, ov::op::internal::RoPE::Config());

        model = std::make_shared<ov::Model>(ov::OutputVector{rope}, ov::ParameterVector{input, rope_input});
        manager.register_pass<IncreasePositionIdsPrecision>();
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{ -1, -1 });
        auto input_convert = std::make_shared<ov::op::v0::Convert>(input, ov::element::i32);
        auto input_unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(input_convert, std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1}));
        auto rotary_embd_const = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1, 64, 1});

        auto input_convert_f32 = std::make_shared<ov::op::v0::Convert>(input_unsqueeze, ov::element::f32);
        auto rotary_embd_const_convert_f32 = std::make_shared<ov::op::v0::Convert>(rotary_embd_const, ov::element::f32);

        auto gemm = std::make_shared<ov::intel_gpu::op::Gemm>(input_convert_f32, rotary_embd_const_convert_f32, std::vector<int64_t>{}, std::vector<int64_t>{}, std::vector<int64_t>{});
        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{gemm, gemm}, 2);

        auto cos = std::make_shared<ov::op::v0::Cos>(concat);
        auto sin = std::make_shared<ov::op::v0::Sin>(concat);

        auto cos_convert = std::make_shared<ov::op::v0::Convert>(cos, ov::element::f16);
        auto sin_convert = std::make_shared<ov::op::v0::Convert>(sin, ov::element::f16);

        auto rope_input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape::dynamic(4));
        auto rope = std::make_shared<ov::op::internal::RoPE>(ov::OutputVector{rope_input, cos_convert, sin_convert}, ov::op::internal::RoPE::Config());

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{rope}, ov::ParameterVector{input, rope_input});
    }
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, IncreasePositionIdsPrecisionWithUnsqueeze) {
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{ -1, -1 });
        auto input_convert = std::make_shared<ov::op::v0::Convert>(input, ov::element::i32);
        auto input_unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(input_convert, std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1}));
        auto input_convert_fp = std::make_shared<ov::op::v0::Convert>(input_unsqueeze, ov::element::f16);
        auto rotary_embd_const = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1, 64, 1});

        auto gemm = std::make_shared<ov::intel_gpu::op::Gemm>(input_convert_fp, rotary_embd_const, std::vector<int64_t>{}, std::vector<int64_t>{}, std::vector<int64_t>{});
        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{gemm, gemm}, 2);

        auto cos = std::make_shared<ov::op::v0::Cos>(concat);
        auto sin = std::make_shared<ov::op::v0::Sin>(concat);

        auto cos_unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(cos, std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1}));
        auto sin_unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(sin, std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1}));

        auto rope_input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape::dynamic(4));
        auto rope = std::make_shared<ov::op::internal::RoPE>(ov::OutputVector{rope_input, cos_unsqueeze, sin_unsqueeze}, ov::op::internal::RoPE::Config());

        model = std::make_shared<ov::Model>(ov::OutputVector{rope}, ov::ParameterVector{input, rope_input});
        manager.register_pass<IncreasePositionIdsPrecision>();
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{ -1, -1 });
        auto input_convert = std::make_shared<ov::op::v0::Convert>(input, ov::element::i32);
        auto input_unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(input_convert, std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1}));
        auto rotary_embd_const = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1, 64, 1});

        auto input_convert_f32 = std::make_shared<ov::op::v0::Convert>(input_unsqueeze, ov::element::f32);
        auto rotary_embd_const_convert_f32 = std::make_shared<ov::op::v0::Convert>(rotary_embd_const, ov::element::f32);

        auto gemm = std::make_shared<ov::intel_gpu::op::Gemm>(input_convert_f32, rotary_embd_const_convert_f32, std::vector<int64_t>{}, std::vector<int64_t>{}, std::vector<int64_t>{});
        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{gemm, gemm}, 2);

        auto cos = std::make_shared<ov::op::v0::Cos>(concat);
        auto sin = std::make_shared<ov::op::v0::Sin>(concat);

        auto cos_convert = std::make_shared<ov::op::v0::Convert>(cos, ov::element::f16);
        auto sin_convert = std::make_shared<ov::op::v0::Convert>(sin, ov::element::f16);

        auto cos_unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(cos_convert, std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1}));
        auto sin_unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(sin_convert, std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1}));

        auto rope_input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape::dynamic(4));
        auto rope = std::make_shared<ov::op::internal::RoPE>(ov::OutputVector{rope_input, cos_unsqueeze, sin_unsqueeze}, ov::op::internal::RoPE::Config());

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{rope}, ov::ParameterVector{input, rope_input});
    }
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, IncreasePositionIdsMatmulWithoutUnsqueeze) {
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{ -1, -1 });
        auto input_convert = std::make_shared<ov::op::v0::Convert>(input, ov::element::i32);
        auto input_unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(input_convert, std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1}));
        auto input_convert_fp = std::make_shared<ov::op::v0::Convert>(input_unsqueeze, ov::element::f16);
        auto rotary_embd_const = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1, 64, 1});

        auto matmul = std::make_shared<ov::op::v0::MatMul>(input_convert_fp, rotary_embd_const);
        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{matmul, matmul}, 2);

        auto cos = std::make_shared<ov::op::v0::Cos>(concat);
        auto sin = std::make_shared<ov::op::v0::Sin>(concat);

        auto rope_input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape::dynamic(4));
        auto rope = std::make_shared<ov::op::internal::RoPE>(ov::OutputVector{rope_input, cos, sin}, ov::op::internal::RoPE::Config());

        model = std::make_shared<ov::Model>(ov::OutputVector{rope}, ov::ParameterVector{input, rope_input});
        manager.register_pass<IncreasePositionIdsPrecision>();
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{ -1, -1 });
        auto input_convert = std::make_shared<ov::op::v0::Convert>(input, ov::element::i32);
        auto input_unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(input_convert, std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1}));
        auto rotary_embd_const = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1, 64, 1});

        auto input_convert_f32 = std::make_shared<ov::op::v0::Convert>(input_unsqueeze, ov::element::f32);
        auto rotary_embd_const_convert_f32 = std::make_shared<ov::op::v0::Convert>(rotary_embd_const, ov::element::f32);

        auto matmul = std::make_shared<ov::op::v0::MatMul>(input_convert_f32, rotary_embd_const_convert_f32);
        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{matmul, matmul}, 2);

        auto cos = std::make_shared<ov::op::v0::Cos>(concat);
        auto sin = std::make_shared<ov::op::v0::Sin>(concat);

        auto cos_convert = std::make_shared<ov::op::v0::Convert>(cos, ov::element::f16);
        auto sin_convert = std::make_shared<ov::op::v0::Convert>(sin, ov::element::f16);

        auto rope_input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape::dynamic(4));
        auto rope = std::make_shared<ov::op::internal::RoPE>(ov::OutputVector{rope_input, cos_convert, sin_convert}, ov::op::internal::RoPE::Config());

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{rope}, ov::ParameterVector{input, rope_input});
    }
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, IncreasePositionIdsReshapeAfterMatmul) {
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{ -1, -1 });
        auto input_convert = std::make_shared<ov::op::v0::Convert>(input, ov::element::i32);
        auto input_unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(input_convert, std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1}));
        auto input_convert_fp = std::make_shared<ov::op::v0::Convert>(input_unsqueeze, ov::element::f16);
        auto rotary_embd_const = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1, 64, 1});
        auto reshape_dims = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{3});

        auto matmul = std::make_shared<ov::op::v0::MatMul>(input_convert_fp, rotary_embd_const);
        auto reshape = std::make_shared<ov::op::v1::Reshape>(matmul, reshape_dims, true);
        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{reshape, reshape}, 2);

        auto cos = std::make_shared<ov::op::v0::Cos>(concat);
        auto sin = std::make_shared<ov::op::v0::Sin>(concat);

        auto rope_input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape::dynamic(4));
        auto rope = std::make_shared<ov::op::internal::RoPE>(ov::OutputVector{rope_input, cos, sin}, ov::op::internal::RoPE::Config());

        model = std::make_shared<ov::Model>(ov::OutputVector{rope}, ov::ParameterVector{input, rope_input, reshape_dims});
        manager.register_pass<IncreasePositionIdsPrecision>();
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{ -1, -1 });
        auto input_convert = std::make_shared<ov::op::v0::Convert>(input, ov::element::i32);
        auto input_unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(input_convert, std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1}));
        auto rotary_embd_const = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1, 64, 1});
        auto reshape_dims = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{3});

        auto input_convert_f32 = std::make_shared<ov::op::v0::Convert>(input_unsqueeze, ov::element::f32);
        auto rotary_embd_const_convert_f32 = std::make_shared<ov::op::v0::Convert>(rotary_embd_const, ov::element::f32);

        auto matmul = std::make_shared<ov::op::v0::MatMul>(input_convert_f32, rotary_embd_const_convert_f32);
        auto reshape = std::make_shared<ov::op::v1::Reshape>(matmul, reshape_dims, true);
        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{reshape, reshape}, 2);

        auto cos = std::make_shared<ov::op::v0::Cos>(concat);
        auto sin = std::make_shared<ov::op::v0::Sin>(concat);

        auto cos_convert = std::make_shared<ov::op::v0::Convert>(cos, ov::element::f16);
        auto sin_convert = std::make_shared<ov::op::v0::Convert>(sin, ov::element::f16);

        auto rope_input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape::dynamic(4));
        auto rope = std::make_shared<ov::op::internal::RoPE>(ov::OutputVector{rope_input, cos_convert, sin_convert}, ov::op::internal::RoPE::Config());

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{rope}, ov::ParameterVector{input, rope_input, reshape_dims});
    }
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, IncreasePositionIdsLongRoPE) {
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{ -1 });
        auto input_convert = std::make_shared<ov::op::v0::Convert>(input, ov::element::i32);
        auto input_unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(input_convert, std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{-1, 1, 1}));
        auto input_convert_fp = std::make_shared<ov::op::v0::Convert>(input_unsqueeze, ov::element::f16);
        auto rotary_embd = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 48, 1});

        auto matmul = std::make_shared<ov::op::v0::MatMul>(rotary_embd, input_convert_fp);
        auto reshape = std::make_shared<ov::op::v1::Reshape>(matmul, std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{0, 1, 48}), true);
        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{reshape, reshape}, 2);

        auto cos = std::make_shared<ov::op::v0::Cos>(concat);
        auto sin = std::make_shared<ov::op::v0::Sin>(concat);

        auto const_scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 1, 1, 1 }, { 1.19043 });
        auto const_scale = std::make_shared<ov::op::v1::Multiply>(cos, const_scale_const);
        auto sin_scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 1, 1, 1 }, { 1.19043 });
        auto sin_scale = std::make_shared<ov::op::v1::Multiply>(sin, sin_scale_const);

        auto cos_unsqueeze = std::make_shared<ov::op::v1::Reshape>(const_scale, std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{-1, 1, 1, 96}), true);
        auto sin_unsqueeze = std::make_shared<ov::op::v1::Reshape>(sin_scale, std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{-1, 1, 1, 96}), true);

        auto rope_input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape::dynamic(4));
        auto rope = std::make_shared<ov::op::internal::RoPE>(ov::OutputVector{rope_input, cos_unsqueeze, sin_unsqueeze}, ov::op::internal::RoPE::Config());

        model = std::make_shared<ov::Model>(ov::OutputVector{rope}, ov::ParameterVector{input, rope_input, rotary_embd});
        manager.register_pass<IncreasePositionIdsPrecision>();
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{ -1 });
        auto input_convert = std::make_shared<ov::op::v0::Convert>(input, ov::element::i32);
        auto input_unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(input_convert, std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{-1, 1, 1}));
        auto input_convert_fp = std::make_shared<ov::op::v0::Convert>(input_unsqueeze, ov::element::f16);
        auto rotary_embd = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 48, 1});

        auto input_convert_f32 = std::make_shared<ov::op::v0::Convert>(input_unsqueeze, ov::element::f32);
        auto rotary_embd_convert_f32 = std::make_shared<ov::op::v0::Convert>(rotary_embd, ov::element::f32);

        auto matmul = std::make_shared<ov::op::v0::MatMul>(rotary_embd_convert_f32, input_convert_f32);
        auto reshape = std::make_shared<ov::op::v1::Reshape>(matmul, std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{0, 1, 48}), true);
        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{reshape, reshape}, 2);

        auto cos = std::make_shared<ov::op::v0::Cos>(concat);
        auto sin = std::make_shared<ov::op::v0::Sin>(concat);

        auto cos_convert = std::make_shared<ov::op::v0::Convert>(cos, ov::element::f16);
        auto sin_convert = std::make_shared<ov::op::v0::Convert>(sin, ov::element::f16);

        auto const_scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 1, 1, 1 }, { 1.19043 });
        auto const_scale = std::make_shared<ov::op::v1::Multiply>(cos_convert, const_scale_const);
        auto sin_scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 1, 1, 1 }, { 1.19043 });
        auto sin_scale = std::make_shared<ov::op::v1::Multiply>(sin_convert, sin_scale_const);

        auto cos_unsqueeze = std::make_shared<ov::op::v1::Reshape>(const_scale, std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{-1, 1, 1, 96}), true);
        auto sin_unsqueeze = std::make_shared<ov::op::v1::Reshape>(sin_scale, std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{-1, 1, 1, 96}), true);

        auto rope_input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape::dynamic(4));
        auto rope = std::make_shared<ov::op::internal::RoPE>(ov::OutputVector{rope_input, cos_unsqueeze, sin_unsqueeze}, ov::op::internal::RoPE::Config());

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{rope}, ov::ParameterVector{input, rope_input, rotary_embd});
    }
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, IncreasePositionIdsSliceGatherUnsqueezeRoPE) {
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{ -1 });
        auto input_convert = std::make_shared<ov::op::v0::Convert>(input, ov::element::i32);
        auto input_unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(input_convert, std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{-1, 1}));
        auto input_convert_fp = std::make_shared<ov::op::v0::Convert>(input_unsqueeze, ov::element::f16);
        auto rotary_embd = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{32, 1});

        auto matmul = std::make_shared<ov::op::v0::MatMul>(rotary_embd, input_convert_fp);
        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{matmul, matmul}, 2);

        auto cos = std::make_shared<ov::op::v0::Cos>(concat);
        auto sin = std::make_shared<ov::op::v0::Sin>(concat);

        auto constant_11 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 1 }, { 1 });
        auto constant_12 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 1 }, { 0 });
        auto constant_13 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 1 }, { 1 });
        auto sin_slice = std::make_shared<ov::op::v1::StridedSlice>(sin, constant_11, constant_12, constant_13, std::vector<int64_t>{}, std::vector<int64_t>{});
        auto sin_gather = std::make_shared<ov::op::v8::Gather>(sin_slice, constant_11, constant_12);
        auto sin_unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(sin_gather, constant_13);

        auto cos_slice = std::make_shared<ov::op::v1::StridedSlice>(cos, constant_11, constant_12, constant_13, std::vector<int64_t>{}, std::vector<int64_t>{});
        auto cos_gather = std::make_shared<ov::op::v8::Gather>(cos_slice, constant_11, constant_12);
        auto cos_unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(cos_gather, constant_13);

        auto rope_input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape::dynamic(4));
        auto rope = std::make_shared<ov::op::internal::RoPE>(ov::OutputVector{rope_input, cos_unsqueeze, sin_unsqueeze}, ov::op::internal::RoPE::Config());

        model = std::make_shared<ov::Model>(ov::OutputVector{rope}, ov::ParameterVector{input, rope_input, rotary_embd});
        manager.register_pass<IncreasePositionIdsPrecision>();
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{ -1 });
        auto input_convert = std::make_shared<ov::op::v0::Convert>(input, ov::element::i32);
        auto input_unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(input_convert, std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{-1, 1}));
        auto input_convert_fp = std::make_shared<ov::op::v0::Convert>(input_unsqueeze, ov::element::f16);
        auto rotary_embd = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{32, 1});

        auto input_convert_f32 = std::make_shared<ov::op::v0::Convert>(input_unsqueeze, ov::element::f32);
        auto rotary_embd_convert_f32 = std::make_shared<ov::op::v0::Convert>(rotary_embd, ov::element::f32);

        auto matmul = std::make_shared<ov::op::v0::MatMul>(rotary_embd_convert_f32, input_convert_f32);
        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{matmul, matmul}, 2);

        auto cos = std::make_shared<ov::op::v0::Cos>(concat);
        auto sin = std::make_shared<ov::op::v0::Sin>(concat);

        auto cos_convert = std::make_shared<ov::op::v0::Convert>(cos, ov::element::f16);
        auto sin_convert = std::make_shared<ov::op::v0::Convert>(sin, ov::element::f16);

        auto constant_11 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 1 }, { 1 });
        auto constant_12 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 1 }, { 0 });
        auto constant_13 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ 1 }, { 1 });
        auto sin_slice = std::make_shared<ov::op::v1::StridedSlice>(sin_convert, constant_11, constant_12, constant_13, std::vector<int64_t>{}, std::vector<int64_t>{});
        auto sin_gather = std::make_shared<ov::op::v8::Gather>(sin_slice, constant_11, constant_12);
        auto sin_unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(sin_gather, constant_13);

        auto cos_slice = std::make_shared<ov::op::v1::StridedSlice>(cos_convert, constant_11, constant_12, constant_13, std::vector<int64_t>{}, std::vector<int64_t>{});
        auto cos_gather = std::make_shared<ov::op::v8::Gather>(cos_slice, constant_11, constant_12);
        auto cos_unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(cos_gather, constant_13);

        auto rope_input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape::dynamic(4));
        auto rope = std::make_shared<ov::op::internal::RoPE>(ov::OutputVector{rope_input, cos_unsqueeze, sin_unsqueeze}, ov::op::internal::RoPE::Config());

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{rope}, ov::ParameterVector{input, rope_input, rotary_embd});
    }
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

static void test_IncreasePositionIdsLTXVideo(std::shared_ptr<ov::Model>& model,
                                             std::shared_ptr<ov::Model>& model_ref,
                                             ov::pass::Manager& manager,
                                             FunctionsComparator& comparator,
                                             bool is_ltx_video) {
    // Test graph matches the post-RoPE-fusion pattern:
    //   Multiply → Add(Const) → Transpose → Reshape → Sin/Cos → T → U → B → GND → T → Concat → RoPE
    {
        auto param_mul_in = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, 3, 64});
        auto param_x = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, 64, 16});

        // Upstream: Multiply → Add(Constant) → Transpose → Reshape
        auto mul_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1, 1, 64}, {3.14f});
        auto multiply = std::make_shared<ov::op::v1::Multiply>(param_mul_in, mul_const);

        auto add_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1, 1, 64}, {1e-6f});
        auto add = std::make_shared<ov::op::v1::Add>(multiply, add_const);

        auto transpose_order = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{3}, {0, 2, 1});
        auto transpose_up = std::make_shared<ov::op::v1::Transpose>(add, transpose_order);

        auto reshape_shape = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{3}, {1, 64, 3});
        auto reshape = std::make_shared<ov::op::v1::Reshape>(transpose_up, reshape_shape, false);

        // Sin path: Sin → Transpose → Unsqueeze → Broadcast → GatherND → Transpose → Concat
        auto sin_node = std::make_shared<ov::op::v0::Sin>(reshape);
        auto sin_t_order = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{3}, {0, 2, 1});
        auto sin_transpose = std::make_shared<ov::op::v1::Transpose>(sin_node, sin_t_order);
        auto sin_unsq_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {0});
        auto sin_unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(sin_transpose, sin_unsq_axis);
        auto sin_bcast_shape = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{4}, {2, 1, 3, 64});
        auto sin_broadcast = std::make_shared<ov::op::v3::Broadcast>(sin_unsqueeze, sin_bcast_shape);
        auto sin_gnd_indices = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1, 8, 3}, {0});
        auto sin_gathernd = std::make_shared<ov::op::v8::GatherND>(sin_broadcast, sin_gnd_indices);
        auto sin_t2_order = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{3}, {0, 2, 1});
        auto sin_transpose2 = std::make_shared<ov::op::v1::Transpose>(sin_gathernd, sin_t2_order);
        auto sin_concat_other = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1, 64, 8}, {0.0f});
        auto sin_concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{sin_concat_other, sin_transpose2}, -1);

        // Cos path: Cos → Transpose → Unsqueeze → Broadcast → GatherND → Transpose → Concat
        auto cos_node = std::make_shared<ov::op::v0::Cos>(reshape);
        auto cos_t_order = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{3}, {0, 2, 1});
        auto cos_transpose = std::make_shared<ov::op::v1::Transpose>(cos_node, cos_t_order);
        auto cos_unsq_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {0});
        auto cos_unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(cos_transpose, cos_unsq_axis);
        auto cos_bcast_shape = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{4}, {2, 1, 3, 64});
        auto cos_broadcast = std::make_shared<ov::op::v3::Broadcast>(cos_unsqueeze, cos_bcast_shape);
        auto cos_gnd_indices = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1, 8, 3}, {0});
        auto cos_gathernd = std::make_shared<ov::op::v8::GatherND>(cos_broadcast, cos_gnd_indices);
        auto cos_t2_order = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{3}, {0, 2, 1});
        auto cos_transpose2 = std::make_shared<ov::op::v1::Transpose>(cos_gathernd, cos_t2_order);
        auto cos_concat_other = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1, 64, 8}, {1.0f});
        auto cos_concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{cos_concat_other, cos_transpose2}, -1);

        ov::op::internal::RoPE::Config rope_config;
        rope_config.is_ltx_video = is_ltx_video;
        rope_config.rotary_ndims = 64;
        auto rope = std::make_shared<ov::op::internal::RoPE>(ov::OutputVector{param_x, cos_concat, sin_concat}, rope_config);

        model = std::make_shared<ov::Model>(ov::OutputVector{rope}, ov::ParameterVector{param_mul_in, param_x});
        manager.register_pass<IncreasePositionIdsPrecision>();
    }
    if (is_ltx_video) {
        auto param_mul_in = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, 3, 64});
        auto param_x = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, 64, 16});

        // Upstream with converts: Convert(f16→f32) before Multiply inputs, Convert on Add constant
        auto convert_mul_in = std::make_shared<ov::op::v0::Convert>(param_mul_in, ov::element::f32);
        auto mul_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1, 1, 64}, {3.14f});
        auto convert_mul_const = ov::util::get_constant_from_source(
            std::make_shared<ov::op::v0::Convert>(mul_const, ov::element::f32));
        auto multiply = std::make_shared<ov::op::v1::Multiply>(convert_mul_in, convert_mul_const);

        auto add_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1, 1, 64}, {1e-6f});
        auto convert_add_const = ov::util::get_constant_from_source(
            std::make_shared<ov::op::v0::Convert>(add_const, ov::element::f32));
        auto add = std::make_shared<ov::op::v1::Add>(multiply, convert_add_const);

        auto transpose_order = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{3}, {0, 2, 1});
        auto transpose_up = std::make_shared<ov::op::v1::Transpose>(add, transpose_order);

        auto reshape_shape = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{3}, {1, 64, 3});
        auto reshape = std::make_shared<ov::op::v1::Reshape>(transpose_up, reshape_shape, false);

        // Sin path with Convert(f32→f16) after Sin
        auto sin_node = std::make_shared<ov::op::v0::Sin>(reshape);
        auto convert_sin = std::make_shared<ov::op::v0::Convert>(sin_node, ov::element::f16);
        auto sin_t_order = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{3}, {0, 2, 1});
        auto sin_transpose = std::make_shared<ov::op::v1::Transpose>(convert_sin, sin_t_order);
        auto sin_unsq_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {0});
        auto sin_unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(sin_transpose, sin_unsq_axis);
        auto sin_bcast_shape = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{4}, {2, 1, 3, 64});
        auto sin_broadcast = std::make_shared<ov::op::v3::Broadcast>(sin_unsqueeze, sin_bcast_shape);
        auto sin_gnd_indices = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1, 8, 3}, {0});
        auto sin_gathernd = std::make_shared<ov::op::v8::GatherND>(sin_broadcast, sin_gnd_indices);
        auto sin_t2_order = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{3}, {0, 2, 1});
        auto sin_transpose2 = std::make_shared<ov::op::v1::Transpose>(sin_gathernd, sin_t2_order);
        auto sin_concat_other = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1, 64, 8}, {0.0f});
        auto sin_concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{sin_concat_other, sin_transpose2}, -1);

        // Cos path with Convert(f32→f16) after Cos
        auto cos_node = std::make_shared<ov::op::v0::Cos>(reshape);
        auto convert_cos = std::make_shared<ov::op::v0::Convert>(cos_node, ov::element::f16);
        auto cos_t_order = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{3}, {0, 2, 1});
        auto cos_transpose = std::make_shared<ov::op::v1::Transpose>(convert_cos, cos_t_order);
        auto cos_unsq_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {0});
        auto cos_unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(cos_transpose, cos_unsq_axis);
        auto cos_bcast_shape = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{4}, {2, 1, 3, 64});
        auto cos_broadcast = std::make_shared<ov::op::v3::Broadcast>(cos_unsqueeze, cos_bcast_shape);
        auto cos_gnd_indices = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1, 8, 3}, {0});
        auto cos_gathernd = std::make_shared<ov::op::v8::GatherND>(cos_broadcast, cos_gnd_indices);
        auto cos_t2_order = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{3}, {0, 2, 1});
        auto cos_transpose2 = std::make_shared<ov::op::v1::Transpose>(cos_gathernd, cos_t2_order);
        auto cos_concat_other = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1, 64, 8}, {1.0f});
        auto cos_concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{cos_concat_other, cos_transpose2}, -1);

        ov::op::internal::RoPE::Config rope_config;
        rope_config.is_ltx_video = true;
        rope_config.rotary_ndims = 64;
        auto rope = std::make_shared<ov::op::internal::RoPE>(ov::OutputVector{param_x, cos_concat, sin_concat}, rope_config);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{rope}, ov::ParameterVector{param_mul_in, param_x});
    }
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, IncreasePositionIdsLTXVideo_T) {
    test_IncreasePositionIdsLTXVideo(model, model_ref, manager, comparator, true);
}

TEST_F(TransformationTestsF, IncreasePositionIdsLTXVideo_F) {
    manager.get_pass_config()->set_callback<IncreasePositionIdsPrecisionForLtxVideo>(
        [](const std::shared_ptr<const ov::Node>&) { return true; });
    test_IncreasePositionIdsLTXVideo(model, model_ref, manager, comparator, false);
}

// model2 topology: Sin/Cos → Gather(repeat_interleave) → Concat
// Replaces the GatherND→Transpose chain used in model1.
static void test_IncreasePositionIdsLTXVideoGather(std::shared_ptr<ov::Model>& model,
                                                   std::shared_ptr<ov::Model>& model_ref,
                                                   ov::pass::Manager& manager,
                                                   FunctionsComparator& comparator,
                                                   bool is_ltx_video) {
    {
        auto param_mul_in = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, 3, 64});
        auto param_x = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, 64, 16});

        // Upstream: Multiply → Add(Constant) → Transpose → Reshape
        auto mul_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1, 1, 64}, {3.14f});
        auto multiply = std::make_shared<ov::op::v1::Multiply>(param_mul_in, mul_const);
        auto add_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1, 1, 64}, {1e-6f});
        auto add = std::make_shared<ov::op::v1::Add>(multiply, add_const);
        auto transpose_order = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{3}, {0, 2, 1});
        auto transpose_up = std::make_shared<ov::op::v1::Transpose>(add, transpose_order);
        auto reshape_shape = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{3}, {1, 64, 3});
        auto reshape = std::make_shared<ov::op::v1::Reshape>(transpose_up, reshape_shape, false);

        // Sin path: Sin → Gather → Concat(Broadcast, Gather)
        auto sin_node = std::make_shared<ov::op::v0::Sin>(reshape);
        auto sin_gather_indices = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{64}, {0});
        auto sin_gather_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {-1});
        auto sin_gather = std::make_shared<ov::op::v8::Gather>(sin_node, sin_gather_indices, sin_gather_axis);
        auto sin_concat_other = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1, 64, 8}, {0.0f});
        auto sin_concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{sin_concat_other, sin_gather}, -1);

        // Cos path: Cos → Gather → Concat(Broadcast, Gather)
        auto cos_node = std::make_shared<ov::op::v0::Cos>(reshape);
        auto cos_gather_indices = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{64}, {0});
        auto cos_gather_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {-1});
        auto cos_gather = std::make_shared<ov::op::v8::Gather>(cos_node, cos_gather_indices, cos_gather_axis);
        auto cos_concat_other = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1, 64, 8}, {1.0f});
        auto cos_concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{cos_concat_other, cos_gather}, -1);

        ov::op::internal::RoPE::Config rope_config;
        rope_config.is_ltx_video = is_ltx_video;
        rope_config.rotary_ndims = 64;
        auto rope = std::make_shared<ov::op::internal::RoPE>(ov::OutputVector{param_x, cos_concat, sin_concat}, rope_config);

        model = std::make_shared<ov::Model>(ov::OutputVector{rope}, ov::ParameterVector{param_mul_in, param_x});
        manager.register_pass<IncreasePositionIdsPrecision>();
    }
    if (is_ltx_video) {
        auto param_mul_in = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, 3, 64});
        auto param_x = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, 64, 16});

        // Upstream with f32 converts inserted by the pass
        auto convert_mul_in = std::make_shared<ov::op::v0::Convert>(param_mul_in, ov::element::f32);
        auto mul_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1, 1, 64}, {3.14f});
        auto convert_mul_const = ov::util::get_constant_from_source(
            std::make_shared<ov::op::v0::Convert>(mul_const, ov::element::f32));
        auto multiply = std::make_shared<ov::op::v1::Multiply>(convert_mul_in, convert_mul_const);
        auto add_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1, 1, 64}, {1e-6f});
        auto convert_add_const = ov::util::get_constant_from_source(
            std::make_shared<ov::op::v0::Convert>(add_const, ov::element::f32));
        auto add = std::make_shared<ov::op::v1::Add>(multiply, convert_add_const);
        auto transpose_order = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{3}, {0, 2, 1});
        auto transpose_up = std::make_shared<ov::op::v1::Transpose>(add, transpose_order);
        auto reshape_shape = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{3}, {1, 64, 3});
        auto reshape = std::make_shared<ov::op::v1::Reshape>(transpose_up, reshape_shape, false);

        // Sin path with Convert(f32→f16) after Sin
        auto sin_node = std::make_shared<ov::op::v0::Sin>(reshape);
        auto convert_sin = std::make_shared<ov::op::v0::Convert>(sin_node, ov::element::f16);
        auto sin_gather_indices = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{64}, {0});
        auto sin_gather_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {-1});
        auto sin_gather = std::make_shared<ov::op::v8::Gather>(convert_sin, sin_gather_indices, sin_gather_axis);
        auto sin_concat_other = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1, 64, 8}, {0.0f});
        auto sin_concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{sin_concat_other, sin_gather}, -1);

        // Cos path with Convert(f32→f16) after Cos
        auto cos_node = std::make_shared<ov::op::v0::Cos>(reshape);
        auto convert_cos = std::make_shared<ov::op::v0::Convert>(cos_node, ov::element::f16);
        auto cos_gather_indices = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{64}, {0});
        auto cos_gather_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {-1});
        auto cos_gather = std::make_shared<ov::op::v8::Gather>(convert_cos, cos_gather_indices, cos_gather_axis);
        auto cos_concat_other = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1, 64, 8}, {1.0f});
        auto cos_concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{cos_concat_other, cos_gather}, -1);

        ov::op::internal::RoPE::Config rope_config;
        rope_config.is_ltx_video = true;
        rope_config.rotary_ndims = 64;
        auto rope = std::make_shared<ov::op::internal::RoPE>(ov::OutputVector{param_x, cos_concat, sin_concat}, rope_config);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{rope}, ov::ParameterVector{param_mul_in, param_x});
    }
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, IncreasePositionIdsLTXVideoGather_T) {
    test_IncreasePositionIdsLTXVideoGather(model, model_ref, manager, comparator, true);
}

TEST_F(TransformationTestsF, IncreasePositionIdsLTXVideoGather_F) {
    manager.get_pass_config()->set_callback<IncreasePositionIdsPrecisionForLtxVideo>(
        [](const std::shared_ptr<const ov::Node>&) { return true; });
    test_IncreasePositionIdsLTXVideoGather(model, model_ref, manager, comparator, false);
}

namespace {

struct LtxAngleTestModel {
    std::shared_ptr<ov::Model> model;
    std::shared_ptr<ov::op::v0::Parameter> coordinates;
    std::shared_ptr<ov::op::v0::Parameter> scale;
    std::shared_ptr<ov::op::v0::Constant> coefficient;
    std::shared_ptr<ov::op::v0::Constant> offset;
    std::shared_ptr<ov::Node> coefficient_downcast;
    std::shared_ptr<ov::Node> scatter;
    std::shared_ptr<ov::Node> sin_consumer;
    std::shared_ptr<ov::Node> cos_consumer;
    std::shared_ptr<ov::Node> rope;
};

LtxAngleTestModel make_ltx_angle_test_model(bool fp32_trig = false,
                                          bool gather_nd = false,
                                          bool original_fp32 = false) {
    LtxAngleTestModel graph;
    auto coordinates = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{2});
    auto scale = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2});
    auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 1, 4});
    const auto value_type = original_fp32 ? ov::element::f32 : ov::element::f16;
    auto positions = std::make_shared<ov::op::v0::Convert>(coordinates, value_type);
    positions->set_friendly_name("ltx_positions");
    auto scale_downcast = std::make_shared<ov::op::v0::Convert>(scale, value_type);
    auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
    auto gather = std::make_shared<ov::op::v8::Gather>(scale_downcast, coordinates, axis);
    gather->set_friendly_name("ltx_scale_gather");
    auto scaled = std::make_shared<ov::op::v1::Multiply>(positions, gather);
    auto divisor = ov::op::v0::Constant::create(value_type, ov::Shape{}, {20});
    auto divided = std::make_shared<ov::op::v1::Divide>(scaled, divisor);
    divided->set_friendly_name("ltx_divide");
    auto start = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
    auto stop = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto step = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto updates = std::make_shared<ov::op::v8::Slice>(divided, start, stop, step);
    auto scatter_indices = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1, 1}, {0});
    graph.scatter = std::make_shared<ov::op::v3::ScatterNDUpdate>(positions, scatter_indices, updates);
    graph.scatter->set_friendly_name("ltx_scatter");
    auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{graph.scatter, graph.scatter}, 0);
    auto split = std::make_shared<ov::op::v1::Split>(concat, axis, 2);
    split->set_friendly_name("ltx_split");
    ov::Output<ov::Node> values = split->output(1);
    const auto trig_type = (fp32_trig || original_fp32) ? ov::element::f32 : ov::element::f16;
    if (!original_fp32) {
        values = std::make_shared<ov::op::v0::Convert>(values, ov::element::f32);
        if (!fp32_trig) {
            values = std::make_shared<ov::op::v0::Convert>(values, ov::element::f16);
        }
    }
    graph.coefficient = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{2}, {31415.927734f});
    graph.offset = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{2}, {-15707.963867f});
    graph.coefficient_downcast = graph.coefficient;
    std::shared_ptr<ov::Node> offset_downcast = graph.offset;
    if (!original_fp32) {
        graph.coefficient_downcast = std::make_shared<ov::op::v0::Convert>(graph.coefficient, trig_type);
        offset_downcast = std::make_shared<ov::op::v0::Convert>(graph.offset, trig_type);
    }
    auto multiply = std::make_shared<ov::op::v1::Multiply>(values, graph.coefficient_downcast);
    auto add = std::make_shared<ov::op::v1::Add>(multiply, offset_downcast);
    auto order = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
    auto transpose = std::make_shared<ov::op::v1::Transpose>(add, order);
    auto shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {1, 1, 2});
    auto reshape = std::make_shared<ov::op::v1::Reshape>(transpose, shape, false);
    auto sin = std::make_shared<ov::op::v0::Sin>(reshape);
    auto cos = std::make_shared<ov::op::v0::Cos>(reshape);
    const auto make_table_path = [&](const std::shared_ptr<ov::Node>& trig,
                                     std::shared_ptr<ov::Node>& consumer) -> std::shared_ptr<ov::Node> {
        if (!gather_nd) {
            auto indices = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {0, 1});
            auto table_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {-1});
            consumer = std::make_shared<ov::op::v8::Gather>(trig, indices, table_axis);
            return consumer;
        }
        auto identity = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {0, 1, 2});
        consumer = std::make_shared<ov::op::v1::Transpose>(trig, identity);
        auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(consumer, axis);
        auto target = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {1, 1, 1, 2});
        auto broadcast = std::make_shared<ov::op::v3::Broadcast>(unsqueeze, target);
        auto indices = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
        auto gather = std::make_shared<ov::op::v8::GatherND>(broadcast, indices);
        return std::make_shared<ov::op::v1::Transpose>(gather, identity);
    };
    const auto sin_values = make_table_path(sin, graph.sin_consumer);
    const auto cos_values = make_table_path(cos, graph.cos_consumer);
    auto padding = ov::op::v0::Constant::create(trig_type, ov::Shape{1, 1, 2}, {0});
    auto sin_table = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{padding, sin_values}, -1);
    auto cos_table = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{padding, cos_values}, -1);
    ov::op::internal::RoPE::Config config;
    config.is_ltx_video = true;
    config.rotary_ndims = 4;
    graph.rope = std::make_shared<ov::op::internal::RoPE>(ov::OutputVector{x, cos_table, sin_table}, config);
    auto second_rope = graph.rope->clone_with_new_inputs(graph.rope->input_values());
    ov::OutputVector outputs{graph.rope, second_rope};
    if (!original_fp32) {
        outputs.push_back(graph.scatter);
        outputs.push_back(graph.coefficient_downcast);
    }
    graph.model = std::make_shared<ov::Model>(outputs, ov::ParameterVector{coordinates, scale, x});
    graph.coordinates = coordinates;
    graph.scale = scale;
    return graph;
}

void run_ltx_angle_pass(const std::shared_ptr<ov::Model>& model) {
    ov::pass::Manager manager;
    manager.register_pass<IncreasePositionIdsPrecisionForLtxVideo>();
    manager.run_passes(model);
}

std::shared_ptr<ov::Node> find_ltx_fp32_node(const std::shared_ptr<ov::Model>& model, const std::string& name) {
    for (const auto& node : model->get_ordered_ops()) {
        if (node->get_friendly_name().find(name + "/fp32_") == 0) {
            return node;
        }
    }
    return nullptr;
}

std::shared_ptr<CollectSinCosConstantsForPrecisionPreservation::ConstantSet> run_ltx_constant_precision_passes(
    const std::shared_ptr<ov::Model>& model,
    bool collect_constants = true) {
    auto constants = std::make_shared<CollectSinCosConstantsForPrecisionPreservation::ConstantSet>();
    ov::pass::Manager manager;
    if (collect_constants) {
        manager.register_pass<CollectSinCosConstantsForPrecisionPreservation>(constants);
    }
    manager.register_pass<ov::pass::KeepConstantsPrecisionAndAddConverts>();
    manager.get_pass_config()->set_callback<ov::pass::KeepConstantsPrecisionAndAddConverts>(
        [constants](const std::shared_ptr<const ov::Node>& node) {
            return skip_constant_precision_preservation(node, *constants);
        });
    manager.register_pass<ov::pass::ConvertPrecision>(
        precisions_map{{ov::element::f32, ov::element::f16}}, type_to_fuse_map{}, true, false, true);
    manager.register_pass<ov::pass::ConstantFolding>();
    manager.run_passes(model);
    return constants;
}

ov::Output<ov::Node> get_ltx_angle(const std::shared_ptr<ov::Node>& consumer) {
    auto trig = consumer->input_value(0).get_node_shared_ptr();
    if (ov::is_type<ov::op::v0::Convert>(trig)) {
        trig = trig->input_value(0).get_node_shared_ptr();
    }
    OPENVINO_ASSERT((ov::is_type_any_of<ov::op::v0::Sin, ov::op::v0::Cos>(trig)));
    return trig->input_value(0);
}

void expect_ltx_angle_region_fp32(const LtxAngleTestModel& graph) {
    const auto sin_angle = get_ltx_angle(graph.sin_consumer);
    const auto cos_angle = get_ltx_angle(graph.cos_consumer);
    EXPECT_EQ(sin_angle, cos_angle);
    ov::OutputVector pending{sin_angle, cos_angle};
    std::unordered_set<const ov::Node*> visited;
    while (!pending.empty()) {
        const auto output = pending.back();
        pending.pop_back();
        ASSERT_EQ(output.get_element_type(), ov::element::f32) << output;
        const auto node = output.get_node_shared_ptr();
        if (!visited.insert(node.get()).second) {
            continue;
        }
        for (const auto& input : node->input_values()) {
            if (input.get_element_type().is_real()) {
                pending.push_back(input);
            }
        }
    }
}

ov::Tensor evaluate_ltx_angle(const LtxAngleTestModel& graph) {
    auto model = std::make_shared<ov::Model>(ov::OutputVector{get_ltx_angle(graph.sin_consumer)},
                                           ov::ParameterVector{graph.coordinates, graph.scale});
    ov::Tensor coordinates(ov::element::i64, ov::Shape{2});
    coordinates.data<int64_t>()[0] = 0;
    coordinates.data<int64_t>()[1] = 1;
    ov::Tensor scale(ov::element::f32, ov::Shape{2});
    scale.data<float>()[0] = 1.f;
    scale.data<float>()[1] = 1.f;
    ov::TensorVector outputs{ov::Tensor(ov::element::f32, ov::Shape{1, 1, 2})};
    OPENVINO_ASSERT(model->evaluate(outputs, ov::TensorVector{coordinates, scale}), "LTX angle evaluation failed");
    return outputs[0];
}

}  // namespace

TEST(IncreasePositionIdsLTXVideoCallbackTest, RemovesFP16RoundTripsFromAnglePaths) {
    struct Case {
        bool fp32_trig;
        bool gather_nd;
    };
    for (const auto& config : {Case{false, false}, Case{false, true}, Case{true, false}}) {
        SCOPED_TRACE(::testing::Message() << "fp32_trig=" << config.fp32_trig << ", gather_nd=" << config.gather_nd);
        auto graph = make_ltx_angle_test_model(config.fp32_trig, config.gather_nd);
        run_ltx_angle_pass(graph.model);
        ASSERT_NO_THROW(graph.model->validate_nodes_and_infer_types());

        auto sin = graph.sin_consumer->input_value(0).get_node_shared_ptr();
        auto cos = graph.cos_consumer->input_value(0).get_node_shared_ptr();
        if (!config.fp32_trig) {
            ASSERT_TRUE(ov::is_type<ov::op::v0::Convert>(sin));
            ASSERT_TRUE(ov::is_type<ov::op::v0::Convert>(cos));
            EXPECT_EQ(sin->get_output_element_type(0), ov::element::f16);
            EXPECT_EQ(cos->get_output_element_type(0), ov::element::f16);
            sin = sin->input_value(0).get_node_shared_ptr();
            cos = cos->input_value(0).get_node_shared_ptr();
        }
        ASSERT_TRUE(ov::is_type<ov::op::v0::Sin>(sin));
        ASSERT_TRUE(ov::is_type<ov::op::v0::Cos>(cos));
        EXPECT_EQ(sin->get_output_element_type(0), ov::element::f32);
        EXPECT_EQ(cos->get_output_element_type(0), ov::element::f32);
        EXPECT_EQ(sin->input_value(0), cos->input_value(0));
        expect_ltx_angle_region_fp32(graph);

        const auto add = sin->input_value(0).get_node()->input_value(0).get_node()->input_value(0).get_node_shared_ptr();
        ASSERT_TRUE(ov::is_type<ov::op::v1::Add>(add));
        EXPECT_EQ(add->input_value(1), graph.offset->output(0));
        const auto multiply = add->input_value(0).get_node_shared_ptr();
        EXPECT_EQ(multiply->input_value(1), graph.coefficient->output(0));
        EXPECT_EQ(multiply->input_value(0).get_index(), 1);
        const auto split = multiply->input_value(0).get_node_shared_ptr();
        ASSERT_TRUE(ov::is_type<ov::op::v1::Split>(split));
        const auto concat = split->input_value(0).get_node_shared_ptr();
        EXPECT_EQ(concat->input_value(0), concat->input_value(1));

        const auto scatter = find_ltx_fp32_node(graph.model, "ltx_scatter");
        ASSERT_NE(scatter, nullptr);
        EXPECT_EQ(concat->input_value(0), scatter->output(0));
        EXPECT_EQ(scatter->get_input_element_type(0), ov::element::f32);
        EXPECT_EQ(scatter->get_input_element_type(2), ov::element::f32);
        EXPECT_EQ(scatter->input_value(1), graph.scatter->input_value(1));
        EXPECT_TRUE(ov::is_type<ov::op::v3::ScatterNDUpdate>(scatter));
        const auto positions = scatter->input_value(0).get_node_shared_ptr();
        ASSERT_TRUE(ov::is_type<ov::op::v0::Convert>(positions));
        EXPECT_EQ(positions->input_value(0), graph.coordinates->output(0));
        EXPECT_EQ(positions->get_output_element_type(0), ov::element::f32);
        const auto gather = find_ltx_fp32_node(graph.model, "ltx_scale_gather");
        ASSERT_NE(gather, nullptr);
        EXPECT_EQ(gather->input_value(0), graph.scale->output(0));
        EXPECT_EQ(gather->input_value(1), graph.coordinates->output(0));
        EXPECT_EQ(gather->get_input_element_type(1), ov::element::i64);
        EXPECT_EQ(gather->get_input_element_type(2), ov::element::i64);
        const auto divide = find_ltx_fp32_node(graph.model, "ltx_divide");
        ASSERT_NE(divide, nullptr);
        EXPECT_EQ(divide->get_input_element_type(0), ov::element::f32);
        EXPECT_EQ(divide->get_input_element_type(1), ov::element::f32);
        ASSERT_TRUE(ov::is_type<ov::op::v0::Constant>(divide->input_value(1).get_node()));
        EXPECT_EQ(graph.model->get_results()[2]->input_value(0), graph.scatter->output(0));
        EXPECT_EQ(graph.model->get_results()[3]->input_value(0), graph.coefficient_downcast->output(0));
        EXPECT_EQ(graph.scatter->get_output_element_type(0), ov::element::f16);
        EXPECT_EQ(graph.coefficient_downcast->get_output_element_type(0),
                  config.fp32_trig ? ov::element::f32 : ov::element::f16);
        EXPECT_EQ(graph.model->get_results()[0]->input_value(0), graph.rope->output(0));
        EXPECT_TRUE(ov::is_type<ov::op::internal::RoPE>(graph.model->get_results()[1]->input_value(0).get_node()));

        const auto nodes = graph.model->get_ordered_ops();
        run_ltx_angle_pass(graph.model);
        EXPECT_EQ(nodes, graph.model->get_ordered_ops());
    }
}

TEST(IncreasePositionIdsLTXVideoCallbackTest, CollectsAndPreservesOriginalConstants) {
    for (const bool gather_nd : {false, true}) {
        SCOPED_TRACE(::testing::Message() << "gather_nd=" << gather_nd);
        auto graph = make_ltx_angle_test_model(false, gather_nd, true);
        auto unrelated_constant = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{2}, {31415.927734f});
        auto unrelated = std::make_shared<ov::op::v1::Multiply>(graph.scale, unrelated_constant);
        auto result = std::make_shared<ov::op::v0::Result>(unrelated);
        graph.model->add_results({result});
        const auto coefficient_values = graph.coefficient->cast_vector<float>();
        const auto offset_values = graph.offset->cast_vector<float>();

        const auto constants = run_ltx_constant_precision_passes(graph.model);
        EXPECT_EQ(constants->count(graph.coefficient.get()), 1);
        EXPECT_EQ(constants->count(graph.offset.get()), 1);
        EXPECT_EQ(constants->count(graph.scatter->input_value(1).get_node()), 0);
        EXPECT_EQ(constants->count(unrelated_constant.get()), 0);
        EXPECT_EQ(graph.coefficient->cast_vector<float>(), coefficient_values);
        EXPECT_EQ(graph.offset->cast_vector<float>(), offset_values);

        const auto angle = get_ltx_angle(graph.sin_consumer);
        ASSERT_EQ(angle.get_element_type(), ov::element::f16);
        const auto add = angle.get_node()->input_value(0).get_node()->input_value(0).get_node_shared_ptr();
        ASSERT_TRUE(ov::is_type<ov::op::v1::Add>(add));
        const auto multiply = add->input_value(0).get_node_shared_ptr();
        ASSERT_TRUE(ov::is_type<ov::op::v1::Multiply>(multiply));
        for (const auto& entry : {std::make_pair(multiply->input_value(1), graph.coefficient),
                                  std::make_pair(add->input_value(1), graph.offset)}) {
            const auto convert = ov::as_type_ptr<ov::op::v0::Convert>(entry.first.get_node_shared_ptr());
            ASSERT_NE(convert, nullptr);
            EXPECT_EQ(convert->get_output_element_type(0), ov::element::f16);
            EXPECT_EQ(convert->input_value(0), entry.second->output(0));
            EXPECT_EQ(entry.second->get_element_type(), ov::element::f32);
        }

        auto unrelated_output = result->input_value(0).get_node_shared_ptr();
        if (ov::is_type<ov::op::v0::Convert>(unrelated_output)) {
            unrelated_output = unrelated_output->input_value(0).get_node_shared_ptr();
        }
        ASSERT_TRUE(ov::is_type<ov::op::v1::Multiply>(unrelated_output));
        const auto rounded = ov::util::get_constant_from_source(unrelated_output->input_value(1));
        ASSERT_NE(rounded, nullptr);
        EXPECT_EQ(rounded->get_element_type(), ov::element::f16);
        EXPECT_EQ(rounded->cast_vector<float>()[0], static_cast<float>(ov::float16(coefficient_values[0])));
        EXPECT_NE(rounded->cast_vector<float>()[0], coefficient_values[0]);
        EXPECT_NO_THROW(graph.model->validate_nodes_and_infer_types());
    }
}

TEST(IncreasePositionIdsLTXVideoCallbackTest, RestoresOriginalFP32Angles) {
    for (const bool gather_nd : {false, true}) {
        for (const bool collect_constants : {false, true}) {
            SCOPED_TRACE(::testing::Message() << "gather_nd=" << gather_nd << ", collector=" << collect_constants);
            auto graph = make_ltx_angle_test_model(false, gather_nd, true);
            const auto reference = evaluate_ltx_angle(graph);
            const auto coefficient_values = graph.coefficient->cast_vector<float>();
            const auto offset_values = graph.offset->cast_vector<float>();

            run_ltx_constant_precision_passes(graph.model, collect_constants);
            run_ltx_angle_pass(graph.model);
            ASSERT_NO_THROW(graph.model->validate_nodes_and_infer_types());
            expect_ltx_angle_region_fp32(graph);
            const auto angle = get_ltx_angle(graph.sin_consumer);
            const auto add = angle.get_node()->input_value(0).get_node()->input_value(0).get_node_shared_ptr();
            ASSERT_TRUE(ov::is_type<ov::op::v1::Add>(add));
            const auto multiply = add->input_value(0).get_node_shared_ptr();
            ASSERT_TRUE(ov::is_type<ov::op::v1::Multiply>(multiply));
            const auto coefficient = ov::as_type_ptr<ov::op::v0::Constant>(multiply->input_value(1).get_node_shared_ptr());
            const auto offset = ov::as_type_ptr<ov::op::v0::Constant>(add->input_value(1).get_node_shared_ptr());
            ASSERT_NE(coefficient, nullptr);
            ASSERT_NE(offset, nullptr);
            EXPECT_EQ(coefficient->get_element_type(), ov::element::f32);
            EXPECT_EQ(offset->get_element_type(), ov::element::f32);
            const auto actual = evaluate_ltx_angle(graph);
            ASSERT_EQ(actual.get_shape(), reference.get_shape());
            if (collect_constants) {
                EXPECT_EQ(coefficient, graph.coefficient);
                EXPECT_EQ(offset, graph.offset);
                EXPECT_EQ(coefficient->cast_vector<float>(), coefficient_values);
                EXPECT_EQ(offset->cast_vector<float>(), offset_values);
                for (size_t i = 0; i < actual.get_size(); ++i) {
                    EXPECT_FLOAT_EQ(actual.data<const float>()[i], reference.data<const float>()[i]);
                }
            } else {
                const auto rounded_coefficient = static_cast<float>(ov::float16(coefficient_values[0]));
                const auto rounded_offset = static_cast<float>(ov::float16(offset_values[0]));
                EXPECT_EQ(coefficient->cast_vector<float>(), std::vector<float>(2, rounded_coefficient));
                EXPECT_EQ(offset->cast_vector<float>(), std::vector<float>(2, rounded_offset));
                EXPECT_FLOAT_EQ(actual.data<const float>()[0], rounded_offset);
                EXPECT_FLOAT_EQ(actual.data<const float>()[1], rounded_coefficient + rounded_offset);
                for (size_t i = 0; i < actual.get_size(); ++i) {
                    EXPECT_GT(std::abs(actual.data<const float>()[i] - reference.data<const float>()[i]), 1.f);
                }
            }
        }
    }
}

TEST_F(TransformationTestsF, IncreasePositionIdsPrecisionForQwen25VL) {
    {
        auto position_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{ 3, -1, -1 });
        auto input_convert = std::make_shared<ov::op::v0::Convert>(position_ids, ov::element::i32);
        auto input_unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(input_convert,
                std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int64_t>{0}));
        auto convert_2 = std::make_shared<ov::op::v0::Convert>(input_unsqueeze, ov::element::f16);

        auto shape_of = std::make_shared<ov::op::v3::ShapeOf>(input_convert, ov::element::i32);
        auto gather_0 = std::make_shared<ov::op::v8::Gather>(shape_of,
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1}, std::vector<int64_t>{1}),
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int64_t>{0}));
        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{
                std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1}, std::vector<int64_t>{1}),
                gather_0,
                std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1}, std::vector<int64_t>{64}),
                std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1}, std::vector<int64_t>{1})}, 0);
        auto broadcast = std::make_shared<ov::op::v3::Broadcast>(
            std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1, 1, 64, 1}), concat);
        auto matmul = std::make_shared<ov::op::v0::MatMul>(broadcast, convert_2);
        auto transpose = std::make_shared<ov::op::v1::Transpose>(matmul,
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{4}, std::vector<int64_t>{0, 1, 3, 2}));
        auto concat_2 = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{transpose, transpose}, 3);

        auto cos = std::make_shared<ov::op::v0::Cos>(concat_2);
        auto variadic_split_cos = std::make_shared<ov::op::v1::VariadicSplit>(cos,
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int64_t>{3}),
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{6}, std::vector<int64_t>{16,24,24,16,24,24}));
        auto gather_cos_0 = std::make_shared<ov::op::v8::Gather>(variadic_split_cos->output(0),
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int64_t>{0}),
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int64_t>{0}));
        auto gather_cos_1 = std::make_shared<ov::op::v8::Gather>(variadic_split_cos->output(1),
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int64_t>{0}),
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int64_t>{0}));
        auto gather_cos_2 = std::make_shared<ov::op::v8::Gather>(variadic_split_cos->output(2),
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int64_t>{0}),
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int64_t>{0}));
        auto gather_cos_3 = std::make_shared<ov::op::v8::Gather>(variadic_split_cos->output(3),
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int64_t>{0}),
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int64_t>{0}));
        auto gather_cos_4 = std::make_shared<ov::op::v8::Gather>(variadic_split_cos->output(4),
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int64_t>{0}),
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int64_t>{0}));
        auto gather_cos_5 = std::make_shared<ov::op::v8::Gather>(variadic_split_cos->output(5),
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int64_t>{0}),
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int64_t>{0}));
        auto concat_cos = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{
            gather_cos_0, gather_cos_1, gather_cos_2, gather_cos_3, gather_cos_4, gather_cos_5}, 2);
        auto unsqueeze_cos = std::make_shared<ov::op::v0::Unsqueeze>(concat_cos,
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int64_t>{0}));

        auto sin = std::make_shared<ov::op::v0::Sin>(concat_2);
        auto variadic_split_sin = std::make_shared<ov::op::v1::VariadicSplit>(sin,
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int64_t>{3}),
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{6}, std::vector<int64_t>{16,24,24,16,24,24}));
        auto gather_sin_0 = std::make_shared<ov::op::v8::Gather>(variadic_split_sin->output(0),
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int64_t>{0}),
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int64_t>{0}));
        auto gather_sin_1 = std::make_shared<ov::op::v8::Gather>(variadic_split_sin->output(1),
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int64_t>{0}),
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int64_t>{0}));
        auto gather_sin_2 = std::make_shared<ov::op::v8::Gather>(variadic_split_sin->output(2),
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int64_t>{0}),
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int64_t>{0}));
        auto gather_sin_3 = std::make_shared<ov::op::v8::Gather>(variadic_split_sin->output(3),
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int64_t>{0}),
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int64_t>{0}));
        auto gather_sin_4 = std::make_shared<ov::op::v8::Gather>(variadic_split_sin->output(4),
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int64_t>{0}),
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int64_t>{0}));
        auto gather_sin_5 = std::make_shared<ov::op::v8::Gather>(variadic_split_sin->output(5),
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int64_t>{0}),
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int64_t>{0}));
        auto concat_sin = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{
            gather_sin_0, gather_sin_1, gather_sin_2, gather_sin_3, gather_sin_4, gather_sin_5}, 2);
        auto unsqueeze_sin = std::make_shared<ov::op::v0::Unsqueeze>(concat_sin,
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int64_t>{0}));

        auto input_2 = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{ -1, -1, 28, 128});
        auto rope = std::make_shared<ov::op::internal::RoPE>(ov::OutputVector{input_2, unsqueeze_cos, unsqueeze_sin},
                ov::op::internal::RoPE::Config());

        model = std::make_shared<ov::Model>(ov::OutputVector{rope}, ov::ParameterVector{position_ids, input_2});
        manager.register_pass<IncreasePositionIdsPrecision>();
    }
    {
        auto position_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{ 3, -1, -1 });
        auto input_convert = std::make_shared<ov::op::v0::Convert>(position_ids, ov::element::i32);
        auto input_unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(input_convert,
                std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int64_t>{0}));
        auto convert_2 = std::make_shared<ov::op::v0::Convert>(input_unsqueeze, ov::element::f32);

        auto shape_of = std::make_shared<ov::op::v3::ShapeOf>(input_convert, ov::element::i32);
        auto gather_0 = std::make_shared<ov::op::v8::Gather>(shape_of,
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1}, std::vector<int64_t>{1}),
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int64_t>{0}));
        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{
                std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1}, std::vector<int64_t>{1}),
                gather_0,
                std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1}, std::vector<int64_t>{64}),
                std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1}, std::vector<int64_t>{1})}, 0);
        auto broadcast = std::make_shared<ov::op::v3::Broadcast>(
            std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1, 1, 64, 1}), concat);
        auto broadcast_to_f32 = std::make_shared<ov::op::v0::Convert>(broadcast, ov::element::f32);
        auto matmul = std::make_shared<ov::op::v0::MatMul>(broadcast_to_f32, convert_2);
        auto transpose = std::make_shared<ov::op::v1::Transpose>(matmul,
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{4}, std::vector<int64_t>{0, 1, 3, 2}));
        auto concat_2 = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{transpose, transpose}, 3);

        auto cos = std::make_shared<ov::op::v0::Cos>(concat_2);
        auto cos_to_f16 = std::make_shared<ov::op::v0::Convert>(cos, ov::element::f16);
        auto variadic_split_cos = std::make_shared<ov::op::v1::VariadicSplit>(cos_to_f16,
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int64_t>{3}),
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{6}, std::vector<int64_t>{16,24,24,16,24,24}));
        auto gather_cos_0 = std::make_shared<ov::op::v8::Gather>(variadic_split_cos->output(0),
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int64_t>{0}),
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int64_t>{0}));
        auto gather_cos_1 = std::make_shared<ov::op::v8::Gather>(variadic_split_cos->output(1),
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int64_t>{0}),
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int64_t>{0}));
        auto gather_cos_2 = std::make_shared<ov::op::v8::Gather>(variadic_split_cos->output(2),
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int64_t>{0}),
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int64_t>{0}));
        auto gather_cos_3 = std::make_shared<ov::op::v8::Gather>(variadic_split_cos->output(3),
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int64_t>{0}),
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int64_t>{0}));
        auto gather_cos_4 = std::make_shared<ov::op::v8::Gather>(variadic_split_cos->output(4),
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int64_t>{0}),
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int64_t>{0}));
        auto gather_cos_5 = std::make_shared<ov::op::v8::Gather>(variadic_split_cos->output(5),
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int64_t>{0}),
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int64_t>{0}));
        auto concat_cos = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{
            gather_cos_0, gather_cos_1, gather_cos_2, gather_cos_3, gather_cos_4, gather_cos_5}, 2);
        auto unsqueeze_cos = std::make_shared<ov::op::v0::Unsqueeze>(concat_cos,
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int64_t>{0}));

        auto sin = std::make_shared<ov::op::v0::Sin>(concat_2);
        auto sin_to_f16 = std::make_shared<ov::op::v0::Convert>(sin, ov::element::f16);
        auto variadic_split_sin = std::make_shared<ov::op::v1::VariadicSplit>(sin_to_f16,
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int64_t>{3}),
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{6}, std::vector<int64_t>{16,24,24,16,24,24}));
        auto gather_sin_0 = std::make_shared<ov::op::v8::Gather>(variadic_split_sin->output(0),
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int64_t>{0}),
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int64_t>{0}));
        auto gather_sin_1 = std::make_shared<ov::op::v8::Gather>(variadic_split_sin->output(1),
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int64_t>{0}),
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int64_t>{0}));
        auto gather_sin_2 = std::make_shared<ov::op::v8::Gather>(variadic_split_sin->output(2),
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int64_t>{0}),
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int64_t>{0}));
        auto gather_sin_3 = std::make_shared<ov::op::v8::Gather>(variadic_split_sin->output(3),
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int64_t>{0}),
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int64_t>{0}));
        auto gather_sin_4 = std::make_shared<ov::op::v8::Gather>(variadic_split_sin->output(4),
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int64_t>{0}),
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int64_t>{0}));
        auto gather_sin_5 = std::make_shared<ov::op::v8::Gather>(variadic_split_sin->output(5),
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int64_t>{0}),
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int64_t>{0}));
        auto concat_sin = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{
            gather_sin_0, gather_sin_1, gather_sin_2, gather_sin_3, gather_sin_4, gather_sin_5}, 2);
        auto unsqueeze_sin = std::make_shared<ov::op::v0::Unsqueeze>(concat_sin,
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int64_t>{0}));

        auto input_2 = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{ -1, -1, 28, 128});
        auto rope = std::make_shared<ov::op::internal::RoPE>(ov::OutputVector{input_2, unsqueeze_cos, unsqueeze_sin},
                    ov::op::internal::RoPE::Config());
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{rope}, ov::ParameterVector{position_ids, input_2});
    }
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

namespace {

// Shared builder for Qwen3-family RoPE precision tests. The `insert_pre_convert` callback
// takes the i32 Convert output of `position_ids` and returns the node whose output feeds the
// f16/f32 Convert on the MatMul path, letting each test inject model-specific pre-Convert ops
// (e.g. Reshape/StridedSlice). When `is_ref` is true, the returned model carries the f16<->f32
// Convert nodes that IncreasePositionIdsPrecision is expected to insert.
template <typename InsertPreConvert>
std::shared_ptr<ov::Model> build_qwen3vl_rope_model(InsertPreConvert&& insert_pre_convert, bool is_ref) {
    auto position_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{3, -1});
    auto input_convert = std::make_shared<ov::op::v0::Convert>(position_ids, ov::element::i32);
    auto input_reshape = insert_pre_convert(input_convert);
    auto convert_2 = std::make_shared<ov::op::v0::Convert>(input_reshape,
                                                           is_ref ? ov::element::f32 : ov::element::f16);

    auto shape_of = std::make_shared<ov::op::v3::ShapeOf>(input_convert, ov::element::i32);
    auto gather_0 = std::make_shared<ov::op::v8::Gather>(shape_of,
        std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1}, std::vector<int64_t>{1}),
        std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int64_t>{0}));
    auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1}, std::vector<int64_t>{1}),
            gather_0,
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1}, std::vector<int64_t>{64}),
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1}, std::vector<int64_t>{1})}, 0);
    auto broadcast = std::make_shared<ov::op::v3::Broadcast>(
        std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1, 1, 64, 1}), concat);

    std::shared_ptr<ov::Node> matmul_lhs = broadcast;
    if (is_ref) {
        matmul_lhs = std::make_shared<ov::op::v0::Convert>(broadcast, ov::element::f32);
    }
    auto matmul = std::make_shared<ov::op::v0::MatMul>(matmul_lhs, convert_2);

    auto reshape_transpose = std::make_shared<ov::op::v1::Reshape>(matmul,
        std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{4}, std::vector<int32_t>{3, -1, 1, 64}), true);
    auto gather_ch0 = std::make_shared<ov::op::v8::Gather>(reshape_transpose,
        std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int64_t>{0}),
        std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int64_t>{0}));

    auto concat_2 = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{gather_ch0, gather_ch0}, 2);

    std::shared_ptr<ov::Node> cos = std::make_shared<ov::op::v0::Cos>(concat_2);
    std::shared_ptr<ov::Node> sin = std::make_shared<ov::op::v0::Sin>(concat_2);
    if (is_ref) {
        cos = std::make_shared<ov::op::v0::Convert>(cos, ov::element::f16);
        sin = std::make_shared<ov::op::v0::Convert>(sin, ov::element::f16);
    }
    auto cos_unsqueeze = std::make_shared<ov::op::v1::Reshape>(cos,
        std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{4}, std::vector<int32_t>{-1, 1, 1, 128}), true);
    auto sin_unsqueeze = std::make_shared<ov::op::v1::Reshape>(sin,
        std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{4}, std::vector<int32_t>{-1, 1, 1, 128}), true);

    auto input_2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 8, -1, 128});
    auto rope = std::make_shared<ov::op::internal::RoPE>(
        ov::OutputVector{input_2, cos_unsqueeze, sin_unsqueeze}, ov::op::internal::RoPE::Config());

    return std::make_shared<ov::Model>(ov::OutputVector{rope}, ov::ParameterVector{position_ids, input_2});
}

}  // namespace

TEST_F(TransformationTestsF, IncreasePositionIdsPrecisionForQwen3VL) {
    // Qwen3-VL pattern: position_ids -> Convert(i64->i32) -> Reshape(unsqueeze) -> Convert(i32->f16)
    //   -> MatMul(Broadcast, Convert) -> Reshape(transpose) -> Gather -> Concat(self,self)
    //   -> Sin/Cos -> Reshape(unsqueeze) -> RoPE
    auto insert_pre_convert = [](const std::shared_ptr<ov::Node>& input_convert) -> std::shared_ptr<ov::Node> {
        // Qwen3-VL uses Reshape instead of Unsqueeze
        return std::make_shared<ov::op::v1::Reshape>(input_convert,
                std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{4}, std::vector<int32_t>{3, -1, 1, 1}), true);
    };
    model = build_qwen3vl_rope_model(insert_pre_convert, /*is_ref=*/false);
    manager.register_pass<IncreasePositionIdsPrecision>();
    model_ref = build_qwen3vl_rope_model(insert_pre_convert, /*is_ref=*/true);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, IncreasePositionIdsPrecisionForQwen35) {
    // Qwen3.5 pattern: identical to Qwen3-VL but with an extra Reshape + StridedSlice inserted
    // between Convert(i64->i32) and the pre-Convert(f16) Reshape.
    auto insert_pre_convert = [](const std::shared_ptr<ov::Node>& input_convert) -> std::shared_ptr<ov::Node> {
        auto input_reshape_0 = std::make_shared<ov::op::v1::Reshape>(input_convert,
                std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{3}, std::vector<int32_t>{4, -1, 1}), true);
        auto begin = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {1});
        auto end = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {0});
        auto stride = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {1});
        auto input_slice_0 = std::make_shared<ov::op::v1::StridedSlice>(input_reshape_0, begin, end, stride,
                                                                       std::vector<int64_t>{}, std::vector<int64_t>{});
        return std::make_shared<ov::op::v1::Reshape>(input_slice_0,
                std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{4}, std::vector<int32_t>{3, -1, 1, 1}), true);
    };
    model = build_qwen3vl_rope_model(insert_pre_convert, /*is_ref=*/false);
    manager.register_pass<IncreasePositionIdsPrecision>();
    model_ref = build_qwen3vl_rope_model(insert_pre_convert, /*is_ref=*/true);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

// Verifies FP32 promotion for the direct MatMul -> Sin/Cos topology.
TEST(IncreasePositionIdsPrecisionTest, DirectMatMulSinCos) {
    auto position_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{1, 4});
    position_ids->set_friendly_name("position_ids");
    auto position_ids_i32 = std::make_shared<ov::op::v0::Convert>(position_ids, ov::element::i32);
    auto position_ids_gather = std::make_shared<ov::op::v8::Gather>(
        position_ids_i32,
        ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0}),
        ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0}));
    auto position_ids_unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(
        position_ids_gather,
        ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {0}));
    auto position_ids_f16 = std::make_shared<ov::op::v0::Convert>(position_ids_unsqueeze, ov::element::f16);
    auto frequencies = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{4, 4});
    auto matmul = std::make_shared<ov::op::v0::MatMul>(position_ids_f16, frequencies);
    auto sin = std::make_shared<ov::op::v0::Sin>(matmul);
    auto cos = std::make_shared<ov::op::v0::Cos>(matmul);
    auto shape_of = std::make_shared<ov::op::v3::ShapeOf>(matmul, ov::element::i32);
    auto model =
        std::make_shared<ov::Model>(ov::OutputVector{sin, cos, shape_of}, ov::ParameterVector{position_ids});

    ov::pass::Manager manager;
    manager.register_pass<IncreasePositionIdsPrecisionForDirectMatMulSinCos>();
    manager.run_passes(model);

    EXPECT_EQ(matmul->get_input_element_type(0), ov::element::f32);
    EXPECT_EQ(matmul->get_input_element_type(1), ov::element::f32);
    EXPECT_EQ(matmul->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(sin->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(cos->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(shape_of->get_output_element_type(0), ov::element::i32);

    for (const auto& trig_node : {std::static_pointer_cast<ov::Node>(sin),
                                  std::static_pointer_cast<ov::Node>(cos)}) {
        ASSERT_EQ(trig_node->get_users().size(), 1);
        const auto restore_precision =
            ov::as_type_ptr<ov::op::v0::Convert>(trig_node->get_users().front());
        ASSERT_NE(restore_precision, nullptr);
        EXPECT_EQ(restore_precision->get_output_element_type(0), ov::element::f16);
    }
}

// Verifies that position_ids are detected when they feed the right-hand side of MatMul.
TEST(IncreasePositionIdsPrecisionTest, DirectMatMulSinCosWithPositionIdsOnRhs) {
    auto position_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{4, 1});
    position_ids->set_friendly_name("position_ids");
    auto position_ids_f16 = std::make_shared<ov::op::v0::Convert>(position_ids, ov::element::f16);
    auto frequencies = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{4, 4});
    auto matmul = std::make_shared<ov::op::v0::MatMul>(frequencies, position_ids_f16);
    auto sin = std::make_shared<ov::op::v0::Sin>(matmul);
    auto cos = std::make_shared<ov::op::v0::Cos>(matmul);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{sin, cos}, ov::ParameterVector{position_ids});

    ov::pass::Manager manager;
    manager.register_pass<IncreasePositionIdsPrecisionForDirectMatMulSinCos>();
    manager.run_passes(model);

    EXPECT_EQ(matmul->get_input_element_type(0), ov::element::f32);
    EXPECT_EQ(matmul->get_input_element_type(1), ov::element::f32);
    EXPECT_EQ(matmul->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(sin->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(cos->get_output_element_type(0), ov::element::f32);
}

// Verifies that a similarly shaped graph is ignored when its input is not position_ids.
TEST(IncreasePositionIdsPrecisionTest, DirectMatMulSinCosIgnoresUnrelatedInput) {
    auto input_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{1, 4});
    input_ids->set_friendly_name("input_ids");
    auto input_ids_f16 = std::make_shared<ov::op::v0::Convert>(input_ids, ov::element::f16);
    auto frequencies = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{4, 4});
    auto matmul = std::make_shared<ov::op::v0::MatMul>(input_ids_f16, frequencies);
    auto sin = std::make_shared<ov::op::v0::Sin>(matmul);
    auto cos = std::make_shared<ov::op::v0::Cos>(matmul);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{sin, cos}, ov::ParameterVector{input_ids});

    ov::pass::Manager manager;
    manager.register_pass<IncreasePositionIdsPrecisionForDirectMatMulSinCos>();
    manager.run_passes(model);

    EXPECT_EQ(matmul->get_input_element_type(0), ov::element::f16);
    EXPECT_EQ(matmul->get_input_element_type(1), ov::element::f16);
    EXPECT_EQ(matmul->get_output_element_type(0), ov::element::f16);
    EXPECT_EQ(sin->get_output_element_type(0), ov::element::f16);
    EXPECT_EQ(cos->get_output_element_type(0), ov::element::f16);
}

// Verifies that promoting MatMul does not change another consumer of a shared Convert.
TEST(IncreasePositionIdsPrecisionTest, DirectMatMulSinCosPreservesSharedConvertConsumer) {
    auto position_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{1, 4});
    position_ids->set_friendly_name("position_ids");
    auto position_ids_f16 = std::make_shared<ov::op::v0::Convert>(position_ids, ov::element::f16);
    auto frequencies = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{4, 4});
    auto matmul = std::make_shared<ov::op::v0::MatMul>(position_ids_f16, frequencies);
    auto sin = std::make_shared<ov::op::v0::Sin>(matmul);
    auto cos = std::make_shared<ov::op::v0::Cos>(matmul);
    auto unrelated = std::make_shared<ov::op::v0::Result>(position_ids_f16);
    auto model =
        std::make_shared<ov::Model>(ov::OutputVector{sin, cos, unrelated}, ov::ParameterVector{position_ids});

    ov::pass::Manager manager;
    manager.register_pass<IncreasePositionIdsPrecisionForDirectMatMulSinCos>();
    manager.run_passes(model);

    EXPECT_EQ(matmul->get_input_element_type(0), ov::element::f32);
    EXPECT_EQ(unrelated->get_input_element_type(0), ov::element::f16);
    EXPECT_EQ(unrelated->input_value(0).get_node_shared_ptr(), position_ids_f16);
}

// Verifies that the transformation is skipped when MatMul has an unrelated value consumer.
TEST(IncreasePositionIdsPrecisionTest, DirectMatMulSinCosIgnoresExtraMatMulConsumer) {
    auto position_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{1, 4});
    position_ids->set_friendly_name("position_ids");
    auto position_ids_f16 = std::make_shared<ov::op::v0::Convert>(position_ids, ov::element::f16);
    auto frequencies = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{4, 4});
    auto matmul = std::make_shared<ov::op::v0::MatMul>(position_ids_f16, frequencies);
    auto sin = std::make_shared<ov::op::v0::Sin>(matmul);
    auto cos = std::make_shared<ov::op::v0::Cos>(matmul);
    auto unrelated = std::make_shared<ov::op::v0::Result>(matmul);
    auto model =
        std::make_shared<ov::Model>(ov::OutputVector{sin, cos, unrelated}, ov::ParameterVector{position_ids});

    ov::pass::Manager manager;
    manager.register_pass<IncreasePositionIdsPrecisionForDirectMatMulSinCos>();
    manager.run_passes(model);

    EXPECT_EQ(matmul->get_input_element_type(0), ov::element::f16);
    EXPECT_EQ(matmul->get_input_element_type(1), ov::element::f16);
    EXPECT_EQ(matmul->get_output_element_type(0), ov::element::f16);
    EXPECT_EQ(unrelated->get_input_element_type(0), ov::element::f16);
}

TEST_F(TransformationTestsF, IncreasePositionIdsPrecisionForGPTOSS) {
    {
        auto position_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{ 3, -1, -1 });
        auto input_convert = std::make_shared<ov::op::v0::Convert>(position_ids, ov::element::i32);
        auto input_unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(input_convert,
                std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int64_t>{0}));
        auto convert_2 = std::make_shared<ov::op::v0::Convert>(input_unsqueeze, ov::element::f16);

        auto shape_of = std::make_shared<ov::op::v3::ShapeOf>(input_convert, ov::element::i32);
        auto gather_0 = std::make_shared<ov::op::v8::Gather>(shape_of,
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1}, std::vector<int64_t>{1}),
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int64_t>{0}));
        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{
                std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1}, std::vector<int64_t>{1}),
                gather_0,
                std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1}, std::vector<int64_t>{64}),
                std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1}, std::vector<int64_t>{1})}, 0);
        auto broadcast = std::make_shared<ov::op::v3::Broadcast>(
            std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1, 1, 64, 1}), concat);
        auto matmul = std::make_shared<ov::op::v0::MatMul>(broadcast, convert_2);
        auto transpose = std::make_shared<ov::op::v1::Transpose>(matmul,
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{4}, std::vector<int64_t>{0, 1, 3, 2}));

        auto cos = std::make_shared<ov::op::v0::Cos>(transpose);
        auto sin = std::make_shared<ov::op::v0::Sin>(transpose);

        auto mul_cos_scale = std::make_shared<ov::op::v1::Multiply>(cos,
            std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1}));
        auto unsqueeze_cos = std::make_shared<ov::op::v0::Unsqueeze>(mul_cos_scale,
            std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{}, std::vector<int64_t>{0}));

        auto mul_sin_scale = std::make_shared<ov::op::v1::Multiply>(sin,
            std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1}));
        auto unsqueeze_sin = std::make_shared<ov::op::v0::Unsqueeze>(mul_sin_scale,
            std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{}, std::vector<int64_t>{0}));

        auto input_2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, -1, 28, 128});
        auto rope = std::make_shared<ov::op::internal::RoPE>(ov::OutputVector{input_2, unsqueeze_cos, unsqueeze_sin},
                ov::op::internal::RoPE::Config());

        model = std::make_shared<ov::Model>(ov::OutputVector{rope}, ov::ParameterVector{position_ids, input_2});
        manager.register_pass<IncreasePositionIdsPrecisionForGPTOSS>();
    }
    {
        auto position_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{ 3, -1, -1 });
        auto input_convert = std::make_shared<ov::op::v0::Convert>(position_ids, ov::element::i32);
        auto input_unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(input_convert,
                std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int64_t>{0}));
        auto convert_2 = std::make_shared<ov::op::v0::Convert>(input_unsqueeze, ov::element::f32);

        auto shape_of = std::make_shared<ov::op::v3::ShapeOf>(input_convert, ov::element::i32);
        auto gather_0 = std::make_shared<ov::op::v8::Gather>(shape_of,
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1}, std::vector<int64_t>{1}),
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int64_t>{0}));
        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{
                std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1}, std::vector<int64_t>{1}),
                gather_0,
                std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1}, std::vector<int64_t>{64}),
                std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1}, std::vector<int64_t>{1})}, 0);
        auto broadcast = std::make_shared<ov::op::v3::Broadcast>(
            std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1, 1, 64, 1}), concat);
        auto convert_broadcast = std::make_shared<ov::op::v0::Convert>(broadcast, ov::element::f32);
        auto matmul = std::make_shared<ov::op::v0::MatMul>(convert_broadcast, convert_2);
        auto transpose = std::make_shared<ov::op::v1::Transpose>(matmul,
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{4}, std::vector<int64_t>{0, 1, 3, 2}));

        auto cos = std::make_shared<ov::op::v0::Cos>(transpose);
        auto sin = std::make_shared<ov::op::v0::Sin>(transpose);

        auto mul_cos_scale = std::make_shared<ov::op::v1::Multiply>(
            std::make_shared<ov::op::v0::Convert>(cos, ov::element::f32),
            std::make_shared<ov::op::v0::Convert>(std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1}), ov::element::f32));
        auto unsqueeze_cos = std::make_shared<ov::op::v0::Unsqueeze>(mul_cos_scale,
            std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{}, std::vector<int64_t>{0}));

        auto mul_sin_scale = std::make_shared<ov::op::v1::Multiply>(
            std::make_shared<ov::op::v0::Convert>(sin, ov::element::f32),
            std::make_shared<ov::op::v0::Convert>(std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1}), ov::element::f32));
        auto unsqueeze_sin = std::make_shared<ov::op::v0::Unsqueeze>(mul_sin_scale,
            std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{}, std::vector<int64_t>{0}));

        auto input_2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, -1, 28, 128});
        auto rope = std::make_shared<ov::op::internal::RoPE>(ov::OutputVector{input_2, unsqueeze_cos, unsqueeze_sin},
                ov::op::internal::RoPE::Config());

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{rope}, ov::ParameterVector{position_ids, input_2});
    }
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST(IncreasePrecisionTest, IncreaseRMSInputPrecisionForQwenVLMerger) {
    // Pattern: Qwen2.5-VL Vision Embeddings Merger last block (blocks.31) + Merger MLP
    // Target: down_proj MatMul -> Add -> aten_add -> RMS -> Reshape -> merger_mlp
    //
    // Transformation inserts Convert(f16->f32) before:
    //   - down_proj MatMul (both inputs: aten_mul and weight)
    //   - down_proj_add (bias input only, since down_proj output is already f32)
    //   - aten_add (attn_add input only, skip index 1 which is down_proj_add output)
    //   - rms_m (weight input only, skip index 0 which is aten_add output)
    // And inserts Convert(f32->f16) after rms_m output

    // Input for attn block
    auto input_hidden = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 1536});
    auto residual = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 1536});

    // attn_proj MatMul (simulating attention output projection)
    auto attn_proj_weight = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1536, 1536});
    auto attn_proj = std::make_shared<ov::op::v0::MatMul>(input_hidden, attn_proj_weight, false, true);

    // attn_proj_add (bias)
    auto attn_proj_bias = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1536});
    auto attn_proj_add = std::make_shared<ov::op::v1::Add>(attn_proj, attn_proj_bias);

    // attn_add (residual connection)
    auto attn_add = std::make_shared<ov::op::v1::Add>(residual, attn_proj_add);

    // rms_post_m (RMS normalization after attention)
    auto rms_post_weight = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1536});
    auto rms_post_m = std::make_shared<ov::op::internal::RMS>(attn_add, rms_post_weight, 1e-6f);

    // MLP: gate_proj
    auto gate_proj_weight = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{4096, 1536});
    auto gate_proj = std::make_shared<ov::op::v0::MatMul>(rms_post_m, gate_proj_weight, false, true);
    auto gate_proj_bias = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{4096});
    auto gate_proj_add = std::make_shared<ov::op::v1::Add>(gate_proj, gate_proj_bias);

    // MLP: up_proj
    auto up_proj_weight = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{4096, 1536});
    auto up_proj = std::make_shared<ov::op::v0::MatMul>(rms_post_m, up_proj_weight, false, true);
    auto up_proj_bias = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{4096});
    auto up_proj_add = std::make_shared<ov::op::v1::Add>(up_proj, up_proj_bias);

    // SwiGLU: Swish(gate) * up
    auto act_silu = std::make_shared<ov::op::v4::Swish>(gate_proj_add);
    auto aten_mul = std::make_shared<ov::op::v1::Multiply>(act_silu, up_proj_add);

    // down_proj
    auto down_proj_weight = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1536, 4096});
    auto down_proj = std::make_shared<ov::op::v0::MatMul>(aten_mul, down_proj_weight, false, true);
    auto down_proj_bias = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1536});
    auto down_proj_add = std::make_shared<ov::op::v1::Add>(down_proj, down_proj_bias);

    // aten_add_node (residual connection with attn output)
    auto aten_add_node = std::make_shared<ov::op::v1::Add>(attn_add, down_proj_add);

    // rms_m (RMS before merger)
    auto rms_weight = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1536});
    auto rms_m = std::make_shared<ov::op::internal::RMS>(aten_add_node, rms_weight, 1e-6f);

    // aten_view (Reshape for merger)
    auto reshape_shape = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{3}, std::vector<int32_t>{-1, 4, 1536});
    auto aten_view = std::make_shared<ov::op::v1::Reshape>(rms_m, reshape_shape, true);

    // merger_mlp0
    auto merger_mlp0_weight = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{3584, 1536});
    auto merger_mlp0 = std::make_shared<ov::op::v0::MatMul>(aten_view, merger_mlp0_weight, false, true);
    auto merger_mlp0_bias = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{3584});
    auto merger_mlp0_add = std::make_shared<ov::op::v1::Add>(merger_mlp0, merger_mlp0_bias);

    // merger_mlp1 (GELU activation)
    auto merger_mlp1_gelu = std::make_shared<ov::op::v7::Gelu>(merger_mlp0_add);

    // merger_mlp2 - Note: pattern expects consumers_count(1), so add a dummy consumer
    auto merger_mlp2_weight = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{3584, 3584});
    auto merger_mlp2 = std::make_shared<ov::op::v0::MatMul>(merger_mlp1_gelu, merger_mlp2_weight, false, true);

    // Add a dummy Add to satisfy consumers_count(1) for merger_mlp2
    auto dummy_bias = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{3584});
    auto merger_mlp2_add = std::make_shared<ov::op::v1::Add>(merger_mlp2, dummy_bias);

    auto test_model = std::make_shared<ov::Model>(ov::OutputVector{merger_mlp2_add}, ov::ParameterVector{input_hidden, residual});

    // Verify: before transformation, down_proj inputs should be f16
    EXPECT_EQ(down_proj->get_input_element_type(0), ov::element::f16);
    EXPECT_EQ(down_proj->get_input_element_type(1), ov::element::f16);
    EXPECT_EQ(rms_m->get_output_element_type(0), ov::element::f16);

    // Run transformation
    ov::pass::Manager manager;
    manager.register_pass<IncreaseRMSInputPrecision>();
    manager.run_passes(test_model);

    // Verify: after transformation, down_proj inputs should be f32 (via Convert nodes)
    EXPECT_EQ(down_proj->get_input_element_type(0), ov::element::f32)
        << "down_proj input 0 should be f32 after transformation";
    EXPECT_EQ(down_proj->get_input_element_type(1), ov::element::f32)
        << "down_proj input 1 should be f32 after transformation";

    // Verify: rms_m output should be f32 (Convert f32->f16 is added after)
    EXPECT_EQ(rms_m->get_output_element_type(0), ov::element::f32)
        << "rms_m output should be f32 after transformation";

    // Verify: Convert nodes are inserted before down_proj inputs
    auto down_proj_input0 = down_proj->get_input_node_shared_ptr(0);
    auto down_proj_input1 = down_proj->get_input_node_shared_ptr(1);
    EXPECT_TRUE(ov::as_type_ptr<ov::op::v0::Convert>(down_proj_input0) != nullptr)
        << "down_proj input 0 should come from Convert node";
    EXPECT_TRUE(ov::as_type_ptr<ov::op::v0::Convert>(down_proj_input1) != nullptr)
        << "down_proj input 1 should come from Convert node";

    // Verify: Convert node is inserted after rms_m output
    EXPECT_EQ(rms_m->get_users().size(), 1u);
    auto rms_user = rms_m->get_users()[0];
    EXPECT_TRUE(ov::as_type_ptr<ov::op::v0::Convert>(rms_user) != nullptr)
        << "rms_m output should feed into Convert node";
}

// Gemma4 pattern (latest): MatMul(f16) -> Reshape(transpose) -> Concat -> Sin/Cos -> Reshape(unsqueeze) -> RoPE(f16 cos/sin)
// The local / global attention RoPEs share a single position_ids Convert, so that Convert has two users.
TEST_F(TransformationTestsF, IncreasePositionIdsPrecisionForGemma4) {
    // Builds one RoPE branch on top of the shared position_ids Convert.
    // theta_idx only makes the frequency constants of the two branches distinct.
    auto make_rope_branch = [](const ov::Output<ov::Node>& pos_ids_fp,
                               const ov::Output<ov::Node>& rope_input,
                               int theta_idx,
                               bool promoted) {
        auto freq_const = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1, 128, 1},
                                                                 std::vector<float>(128, static_cast<float>(theta_idx + 1)));
        auto shape_const = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{3}, std::vector<int32_t>{1, 128, 1});
        ov::Output<ov::Node> matmul_lhs = std::make_shared<ov::op::v3::Broadcast>(freq_const, shape_const);
        if (promoted)
            matmul_lhs = std::make_shared<ov::op::v0::Convert>(matmul_lhs, ov::element::f32);

        auto matmul = std::make_shared<ov::op::v0::MatMul>(matmul_lhs, pos_ids_fp);

        // Reshape (acts as transpose) -> Concat -> Sin/Cos
        auto reshape_tr_const = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{3}, std::vector<int32_t>{-1, 1, 128});
        auto reshape_transpose = std::make_shared<ov::op::v1::Reshape>(matmul, reshape_tr_const, false);
        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{reshape_transpose, reshape_transpose}, 2);
        ov::Output<ov::Node> cos = std::make_shared<ov::op::v0::Cos>(concat);
        ov::Output<ov::Node> sin = std::make_shared<ov::op::v0::Sin>(concat);

        // RoPE stays in f16, so the promoted branch needs restore converts
        if (promoted) {
            cos = std::make_shared<ov::op::v0::Convert>(cos, ov::element::f16);
            sin = std::make_shared<ov::op::v0::Convert>(sin, ov::element::f16);
        }

        // Reshape (acts as unsqueeze) -> RoPE
        auto reshape_cos_const = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{4}, std::vector<int32_t>{-1, 1, 1, 256});
        auto cos_reshape = std::make_shared<ov::op::v1::Reshape>(cos, reshape_cos_const, false);
        auto reshape_sin_const = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{4}, std::vector<int32_t>{-1, 1, 1, 256});
        auto sin_reshape = std::make_shared<ov::op::v1::Reshape>(sin, reshape_sin_const, false);

        return std::make_shared<ov::op::internal::RoPE>(ov::OutputVector{rope_input, cos_reshape, sin_reshape},
                                                        ov::op::internal::RoPE::Config());
    };

    {
        // position_ids(i64) -> Convert(i32) -> Reshape[?,1,1] -> Convert(f16) -> { MatMul_local, MatMul_global }
        auto position_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{-1});
        auto convert_i32 = std::make_shared<ov::op::v0::Convert>(position_ids, ov::element::i32);
        auto reshape_const = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{3}, std::vector<int32_t>{-1, 1, 1});
        auto reshape_pos = std::make_shared<ov::op::v1::Reshape>(convert_i32, reshape_const, false);
        auto convert_f16 = std::make_shared<ov::op::v0::Convert>(reshape_pos, ov::element::f16);

        auto rope_input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape::dynamic(4));
        auto rope_local = make_rope_branch(convert_f16, rope_input, 0, false);
        auto rope_global = make_rope_branch(convert_f16, rope_input, 1, false);

        model = std::make_shared<ov::Model>(ov::OutputVector{rope_local, rope_global}, ov::ParameterVector{position_ids, rope_input});
        manager.register_pass<IncreasePositionIdsPrecision>();
    }
    {
        auto position_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{-1});
        auto convert_i32 = std::make_shared<ov::op::v0::Convert>(position_ids, ov::element::i32);
        auto reshape_const = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{3}, std::vector<int32_t>{-1, 1, 1});
        auto reshape_pos = std::make_shared<ov::op::v1::Reshape>(convert_i32, reshape_const, false);
        auto convert_f32_local = std::make_shared<ov::op::v0::Convert>(reshape_pos, ov::element::f32);
        auto convert_f32_global = std::make_shared<ov::op::v0::Convert>(reshape_pos, ov::element::f32);

        auto rope_input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape::dynamic(4));
        auto rope_local = make_rope_branch(convert_f32_local, rope_input, 0, true);
        auto rope_global = make_rope_branch(convert_f32_global, rope_input, 1, true);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{rope_local, rope_global}, ov::ParameterVector{position_ids, rope_input});
    }
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}