// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>

#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/test_common.hpp"
#include "transformations/cpu_opset/x64/pass/qkv_proj_fusion.hpp"
#include "transformations/cpu_opset/x64/op/qkv_proj.hpp"
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

// Combined q/k/v weight followed by a VariadicSplit (QKVProjFusionPass2).
std::shared_ptr<ov::Model> make_qkv_combined_model(const PartialShape& input_shape,
                                                   int64_t split_axis,
                                                   const element::Type& lengths_type,
                                                   const std::vector<size_t>& proj_sizes,
                                                   size_t hidden_size) {
    auto input_param = std::make_shared<v0::Parameter>(element::f32, input_shape);
    const auto total = proj_sizes[0] + proj_sizes[1] + proj_sizes[2];
    auto qkv_proj_weight_const = std::make_shared<v0::Constant>(element::f16, Shape{total, hidden_size});
    auto qkv_proj_weight_cvt = std::make_shared<v0::Convert>(qkv_proj_weight_const, element::f32);
    auto qkv_proj = std::make_shared<v0::MatMul>(input_param, qkv_proj_weight_cvt, false, true);
    auto axis = v0::Constant::create(element::i64, Shape{}, {split_axis});
    auto lengths = v0::Constant::create(lengths_type, Shape{3}, proj_sizes);
    auto qkv_split = std::make_shared<v1::VariadicSplit>(qkv_proj, axis, lengths);
    return std::make_shared<ov::Model>(qkv_split->outputs(), ParameterVector{input_param});
}

std::shared_ptr<ov::Model> make_qkv_combined_ref(const PartialShape& input_shape,
                                                 const std::vector<size_t>& proj_sizes,
                                                 size_t hidden_size) {
    auto input_param = std::make_shared<v0::Parameter>(element::f32, input_shape);
    const auto total = proj_sizes[0] + proj_sizes[1] + proj_sizes[2];
    auto qkv_proj_weight_const = std::make_shared<v0::Constant>(element::f16, Shape{total, hidden_size});
    intel_cpu::QKVProjectionNode::Config config{false,
                                                static_cast<int>(hidden_size),
                                                static_cast<int>(proj_sizes[0]),
                                                static_cast<int>(proj_sizes[1]),
                                                static_cast<int>(proj_sizes[2]),
                                                true};
    auto qkv_proj = std::make_shared<intel_cpu::QKVProjectionNode>(
        OutputVector{input_param, qkv_proj_weight_const, qkv_proj_weight_const, qkv_proj_weight_const},
        config);
    return std::make_shared<ov::Model>(qkv_proj->outputs(), ParameterVector{input_param});
}

void register_qkv_fusion(ov::pass::Manager& manager) {
    manager.register_pass<ov::intel_cpu::QKVProjFusion>();
    manager.get_pass_config()->set_callback<ov::intel_cpu::QKVProjFusionPass2>(
        [](const std::shared_ptr<const ov::Node>&) -> bool {
            return true;
        });
}

}  // namespace

// [tokens, hidden] input, GQA sizes, positive split axis and i64 lengths
TEST_F(TransformationTestsF, QKVProjFusion2Rank2GQATest) {
    disable_rt_info_check();
    disable_result_friendly_names_check();
    const size_t hidden_size = 2048;
    const std::vector<size_t> proj_sizes{2048, 256, 256};
    const PartialShape input_shape{-1, static_cast<int64_t>(hidden_size)};

    model = make_qkv_combined_model(input_shape, 1, element::i64, proj_sizes, hidden_size);
    register_qkv_fusion(manager);
    model_ref = make_qkv_combined_ref(input_shape, proj_sizes, hidden_size);
}

TEST_F(TransformationTestsF, QKVProjFusion2Rank3Test) {
    disable_rt_info_check();
    disable_result_friendly_names_check();
    const size_t hidden_size = 2048;
    const std::vector<size_t> proj_sizes{2048, 2048, 2048};
    const PartialShape input_shape{-1, -1, static_cast<int64_t>(hidden_size)};

    model = make_qkv_combined_model(input_shape, -1, element::i32, proj_sizes, hidden_size);
    register_qkv_fusion(manager);
    model_ref = make_qkv_combined_ref(input_shape, proj_sizes, hidden_size);
}

// A split that does not cut the last (output channel) dim must not fuse.
TEST_F(TransformationTestsF, QKVProjFusion2NotLastAxisTest) {
    const size_t hidden_size = 2048;
    model = make_qkv_combined_model(PartialShape{-1, 3, static_cast<int64_t>(hidden_size)},
                                    1,
                                    element::i64,
                                    {1, 1, 1},
                                    hidden_size);
    register_qkv_fusion(manager);
}
