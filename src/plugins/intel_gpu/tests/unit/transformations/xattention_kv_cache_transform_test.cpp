// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "../../../src/plugin/transformations_pipeline.cpp"
#include "intel_gpu/op/fully_connected_compressed.hpp"
#include "intel_gpu/op/placeholder.hpp"
#include "intel_gpu/plugin/remote_context.hpp"
#include "intel_gpu/plugin/transformations_pipeline.hpp"
#include "intel_gpu/runtime/execution_config.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "ov_ops/dynamic_quantize.hpp"
#include "test_utils.h"

using namespace tests;

namespace ov::test::intel_gpu {

namespace v0 = ov::op::v0;

namespace {

std::shared_ptr<ov::Model> create_xattention_paged_attention_model() {
    constexpr int64_t num_heads = 2;
    constexpr int64_t head_size = 64;

    auto query = std::make_shared<v0::Parameter>(element::f16, PartialShape{-1, num_heads * head_size});
    auto key = std::make_shared<v0::Parameter>(element::f16, PartialShape{-1, num_heads * head_size});
    auto value = std::make_shared<v0::Parameter>(element::f16, PartialShape{-1, num_heads * head_size});
    // TransformationsPipeline materializes XAttention KV cache as a 4D tensor:
    // [num_blocks, num_kv_heads, block_size, adjusted_head_size].
    auto key_cache = std::make_shared<v0::Parameter>(element::dynamic, PartialShape::dynamic(4));
    auto value_cache = std::make_shared<v0::Parameter>(element::dynamic, PartialShape::dynamic(4));
    auto past_lens = v0::Constant::create(element::i32, Shape{1}, {0});
    auto subsequence_begins = v0::Constant::create(element::i32, Shape{2}, {0, 1});
    auto block_indices = v0::Constant::create(element::i32, Shape{4}, {0, 1, 2, 3});
    auto block_indices_begins = v0::Constant::create(element::i32, Shape{2}, {0, 4});
    auto scale = v0::Constant::create(element::f32, Shape{}, {1.0f});
    auto sliding_window = v0::Constant::create(element::i32, Shape{}, {0});
    auto alibi_slopes = v0::Constant::create(element::f16, Shape{0}, {});
    auto max_context_len = v0::Constant::create(element::i32, Shape{}, {256});
    auto score_aggregation_window = v0::Constant::create(element::i32, Shape{1}, {1});
    auto rotated_block_indices = v0::Constant::create(element::i32, Shape{0}, {});
    auto rotation_deltas = v0::Constant::create(element::i32, Shape{0, 1}, {});
    auto rotation_trig_lut = std::make_shared<v0::Parameter>(element::f16, PartialShape{-1, head_size});
    auto xattention_threshold = v0::Constant::create(element::f32, Shape{1}, {0.0f});
    auto xattention_block_size = std::make_shared<v0::Parameter>(element::i32, Shape{});
    auto xattention_stride = v0::Constant::create(element::i32, Shape{}, {16});
    auto sinks = v0::Constant::create(element::f16, Shape{0, 0, 0, 0}, {});
    auto adaptive_rkv_start_size = v0::Constant::create(element::i32, Shape{}, {0});
    auto adaptive_rkv_evictable_sizes = v0::Constant::create(element::i32, Shape{1}, {0});
    auto adaptive_rkv_diversity_block_set_indices = v0::Constant::create(element::i32, Shape{0}, {});
    auto adaptive_rkv_diversity_block_set_indices_begins = v0::Constant::create(element::i32, Shape{2}, {0, 0});
    auto token_type_ids = v0::Constant::create(element::i32, Shape{0}, {});
    auto qq_bias = v0::Constant::create(element::u8, Shape{0}, {});
    auto qq_bias_begins = v0::Constant::create(element::i32, Shape{0}, {});

    key_cache->set_friendly_name("key_cache");
    value_cache->set_friendly_name("value_cache");
    xattention_block_size->set_friendly_name("xattention_block_size");

    auto pa = std::make_shared<op::PagedAttentionExtension>(OutputVector{query,
                                                                         key,
                                                                         value,
                                                                         key_cache,
                                                                         value_cache,
                                                                         past_lens,
                                                                         subsequence_begins,
                                                                         block_indices,
                                                                         block_indices_begins,
                                                                         scale,
                                                                         sliding_window,
                                                                         alibi_slopes,
                                                                         max_context_len,
                                                                         score_aggregation_window,
                                                                         rotated_block_indices,
                                                                         rotation_deltas,
                                                                         rotation_trig_lut,
                                                                         xattention_threshold,
                                                                         xattention_block_size,
                                                                         xattention_stride,
                                                                         sinks,
                                                                         adaptive_rkv_start_size,
                                                                         adaptive_rkv_evictable_sizes,
                                                                         adaptive_rkv_diversity_block_set_indices,
                                                                         adaptive_rkv_diversity_block_set_indices_begins,
                                                                         token_type_ids,
                                                                         qq_bias,
                                                                         qq_bias_begins});
    pa->get_rt_info()["num_k_heads"] = num_heads;
    pa->get_rt_info()["k_head_size"] = head_size;
    pa->get_rt_info()["num_v_heads"] = num_heads;
    pa->get_rt_info()["v_head_size"] = head_size;

    auto model = std::make_shared<Model>(OutputVector{pa},
                                         ParameterVector{query,
                                                         key,
                                                         value,
                                                         key_cache,
                                                         value_cache,
                                                         rotation_trig_lut,
                                                         xattention_block_size});
    model->set_rt_info("f16", "runtime_options", ov::hint::kv_cache_precision.name());
    return model;
}

std::shared_ptr<v0::Parameter> find_parameter_by_name(const std::shared_ptr<const ov::Model>& model,
                                                      const std::string& friendly_name) {
    for (const auto& parameter : model->get_parameters()) {
        if (parameter->get_friendly_name() == friendly_name) {
            return parameter;
        }
    }
    return nullptr;
}

std::shared_ptr<ov::Model> create_compressed_fc_model(bool dynamic_data,
                                                      bool transpose_b,
                                                      size_t output_features) {
    constexpr size_t input_features = 128;
    constexpr size_t group_size = 64;
    auto data_shape = dynamic_data ? PartialShape{-1, -1, input_features} : PartialShape{1, 4, input_features};
    auto weight_shape = transpose_b ? Shape{output_features, input_features}
                                    : Shape{input_features, output_features};
    auto scale_shape = transpose_b ? Shape{output_features, input_features / group_size}
                                   : Shape{input_features / group_size, output_features};

    auto data = std::make_shared<v0::Parameter>(element::f16, data_shape);
    auto weights = std::make_shared<v0::Constant>(element::u4, weight_shape);
    auto scale = std::make_shared<v0::Constant>(element::f16, scale_shape);
    auto bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
    auto fc = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(data,
                                                                            weights,
                                                                            bias,
                                                                            scale,
                                                                            element::f16,
                                                                            transpose_b);
    return std::make_shared<ov::Model>(OutputVector{fc}, ParameterVector{data});
}

bool has_node_type(const std::shared_ptr<const ov::Model>& model, const std::string& type_name) {
    const auto ordered_ops = model->get_ordered_ops();
    return std::any_of(ordered_ops.begin(), ordered_ops.end(), [&type_name](const auto& node) {
        return std::string(node->get_type_name()) == type_name;
    });
}

}  // namespace

using DynamicQuantizeParams = std::tuple<bool, bool, size_t, bool>;

class DynamicQuantizeTransformPipelineTest : public testing::TestWithParam<DynamicQuantizeParams> {};

TEST_P(DynamicQuantizeTransformPipelineTest, HandlesWeightLayout) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad) {
        GTEST_SKIP() << "Dynamic quantization requires IMMAD support";
    }

    const auto& [dynamic_data, transpose_b, output_features, expected_dynamic_quantize] = GetParam();
    auto context = std::make_shared<ov::intel_gpu::RemoteContextImpl>("GPU", std::vector<cldnn::device::ptr>{engine.get_device()});
    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::use_onednn(true));
    config.set_user_property(ov::hint::dynamic_quantization_group_size(64));

    auto model = create_compressed_fc_model(dynamic_data, transpose_b, output_features);
    config.finalize(context.get(), model.get());

    ov::intel_gpu::TransformationsPipeline pipeline(config, context);
    pipeline.apply(model);

    EXPECT_TRUE(has_node_type(model, "FullyConnectedCompressed"));
    EXPECT_EQ(has_node_type(model, "DynamicQuantize"), expected_dynamic_quantize);
}

INSTANTIATE_TEST_SUITE_P(smoke_DynamicQuantization,
                         DynamicQuantizeTransformPipelineTest,
                         testing::Values(DynamicQuantizeParams{false, true, 1, false},
                                         DynamicQuantizeParams{false, false, 1, false},
                                         DynamicQuantizeParams{true, true, 1, false},
                                         DynamicQuantizeParams{true, false, 1, false},
                                         DynamicQuantizeParams{true, true, 16, true},
                                         DynamicQuantizeParams{true, false, 16, true}));

TEST(XAttentionTransformPipelineTest, NormalizesByTokenFp16RtInfoToCompressedCacheLayout) {
    auto& engine = get_test_engine();
    auto context = std::make_shared<ov::intel_gpu::RemoteContextImpl>("GPU", std::vector<cldnn::device::ptr>{engine.get_device()});
    auto model = create_xattention_paged_attention_model();

    auto config = get_test_default_config(engine);
    config.set_user_property({ov::internal::key_cache_quant_mode(ov::internal::CacheQuantMode::BY_TOKEN)});
    EXPECT_EQ(config.get_key_cache_quant_mode(), ov::internal::CacheQuantMode::BY_TOKEN);
    config.finalize(context.get(), model.get());
    EXPECT_EQ(config.get_key_cache_quant_mode(), ov::internal::CacheQuantMode::BY_TOKEN);

    try {
        ov::intel_gpu::TransformationsPipeline pipeline(config, context);
        pipeline.apply(model);
    } catch (const std::exception& e) {
        const std::string message = e.what();
        if (message.find("XAttention is not supported by your current GPU architecture or IGC version") != std::string::npos) {
            GTEST_SKIP() << message;
        }
        throw;
    }

    auto key_cache = find_parameter_by_name(model, "key_cache");
    auto value_cache = find_parameter_by_name(model, "value_cache");

    ASSERT_NE(key_cache, nullptr);
    ASSERT_NE(value_cache, nullptr);

    EXPECT_EQ(key_cache->get_element_type(), ov::element::i8);
    EXPECT_EQ(value_cache->get_element_type(), ov::element::i8);

    const auto key_shape = key_cache->get_partial_shape();
    const auto value_shape = value_cache->get_partial_shape();
    ASSERT_TRUE(key_shape.rank().is_static());
    ASSERT_TRUE(value_shape.rank().is_static());
    // 4 means the XAttention cache layout stays 4D after conversion:
    // [num_blocks, num_kv_heads, block_size, adjusted_head_size].
    ASSERT_EQ(key_shape.rank().get_length(), 4);
    ASSERT_EQ(value_shape.rank().get_length(), 4);

    EXPECT_TRUE(key_shape[0].is_dynamic());
    EXPECT_TRUE(value_shape[0].is_dynamic());
    // 2 is the number of KV heads propagated from rt_info in this synthetic model.
    EXPECT_EQ(key_shape[1].get_length(), 2);
    EXPECT_EQ(value_shape[1].get_length(), 2);
    // 256 is the dedicated XAttention cache block size on GPU.
    EXPECT_EQ(key_shape[2].get_length(), 256);
    EXPECT_EQ(value_shape[2].get_length(), 256);
    // 68 = 64 head elements + 4 extra i8 BY_TOKEN quantization bytes.
    // Those 4 bytes store per-token scale and zero-point as two fp16 values.
    EXPECT_EQ(key_shape[3].get_length(), 68);
    EXPECT_EQ(value_shape[3].get_length(), 68);
}

class NormalizeMoEOutputReshapeTest : public testing::TestWithParam<std::tuple<bool, bool, int>> {};

TEST_P(NormalizeMoEOutputReshapeTest, ResolvesAxisCopies) {
    const auto [compressed, swapped, target_kind] = GetParam();
    const auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, -1, 8});
    const auto indices = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1, 2});
    const auto weights = ov::op::v0::Constant::create(compressed ? ov::element::u8 : ov::element::f16, ov::Shape{3, 8, 8}, {1});
    std::shared_ptr<ov::Node> gemm;
    if (compressed) {
        gemm = std::make_shared<ov::op::internal::GatherMatmulCompressed>(input,
                                                                          weights,
                                                                          indices,
                                                                          ov::op::v0::Constant::create(ov::element::f16, ov::Shape{0}, std::vector<float>{}),
                                                                          ov::op::v0::Constant::create(ov::element::f16, ov::Shape{3, 8, 1}, {1}),
                                                                          ov::op::v0::Constant::create(ov::element::u8, ov::Shape{3, 8, 1}, {0}));
    } else {
        gemm = std::make_shared<ov::op::internal::GatherMatmul>(input, weights, indices);
    }
    const auto routing = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{2, -1, 1});
    auto product = std::make_shared<ov::op::v1::Multiply>(swapped ? routing : gemm, swapped ? gemm : routing);
    if (target_kind == 7) {
        product = std::make_shared<ov::op::v1::Multiply>(routing, input);
    }
    const auto reduction = std::make_shared<ov::op::v1::ReduceSum>(product, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0}), false);
    ov::ParameterVector parameters{input, indices, routing};
    ov::Output<ov::Node> target = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {0, 8});
    if (target_kind == 1) {
        target = std::make_shared<ov::op::v8::Slice>(ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {3, 0, 8}),
                                                     ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1}),
                                                     ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {3}),
                                                     ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1}));
    } else if (target_kind == 2) {
        const auto shape_parameter = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{2});
        target = shape_parameter;
        parameters.push_back(shape_parameter);
    } else if (target_kind >= 3 && target_kind <= 5) {
        const int64_t second = target_kind == 3 ? -1 : target_kind == 4 ? 0 : 4;
        target = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{0, second});
    } else if (target_kind == 8) {
        target = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {-1, 0});
    } else if (target_kind == 9) {
        target = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {-1, 8});
    }
    const auto reshape = std::make_shared<ov::op::v1::Reshape>(reduction, target, target_kind != 6);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{reshape}, parameters);
    const auto original_shape = reshape->get_output_partial_shape(0);
    ov::pass::Manager manager;
    manager.register_pass<ov::intel_gpu::NormalizeMoEOutputReshape>();
    manager.run_passes(model);
    const bool normalized = target_kind <= 5 || target_kind == 8;
    EXPECT_EQ(reshape->get_special_zero(), normalized ? false : target_kind != 6);
    if (target_kind == 0 || target_kind == 1 || target_kind == 4 || target_kind == 8) {
        const auto value = ov::as_type_ptr<ov::op::v0::Constant>(reshape->get_input_node_shared_ptr(1));
        ASSERT_NE(value, nullptr);
        EXPECT_EQ(value->cast_vector<int64_t>(), (std::vector<int64_t>{-1, 8}));
        EXPECT_EQ(reshape->get_output_partial_shape(0), original_shape);
    } else if (normalized) {
        const auto select = ov::as_type_ptr<ov::op::v1::Select>(reshape->get_input_node_shared_ptr(1));
        ASSERT_NE(select, nullptr);
        EXPECT_EQ(select->input_value(2), target);
        ov::ParameterVector shape_parameters{indices, routing};
        ov::TensorVector inputs{ov::Tensor(ov::element::i32, {4, 2}), ov::Tensor(ov::element::f16, {2, 4, 1})};
        if (target_kind == 2) {
            shape_parameters.push_back(parameters.back());
            inputs.emplace_back(ov::element::i64, ov::Shape{2});
            inputs.back().data<int64_t>()[0] = 0;
            inputs.back().data<int64_t>()[1] = 8;
        }
        const auto shape_model = std::make_shared<ov::Model>(ov::OutputVector{select}, shape_parameters);
        for (const auto& node : shape_model->get_ordered_ops()) {
            EXPECT_FALSE(ov::is_type<ov::op::internal::GatherMatmul>(node));
            EXPECT_FALSE(ov::is_type<ov::op::internal::GatherMatmulCompressed>(node));
        }
        ov::TensorVector outputs{ov::Tensor(ov::element::i64, {2})};
        ASSERT_TRUE(shape_model->evaluate(outputs, inputs));
        EXPECT_EQ(outputs[0].data<int64_t>()[0], 4);
        EXPECT_EQ(outputs[0].data<int64_t>()[1], target_kind == 3 ? -1 : target_kind == 5 ? 4 : 8);
    } else {
        EXPECT_EQ(reshape->input_value(1), target);
    }
}

INSTANTIATE_TEST_SUITE_P(smoke, NormalizeMoEOutputReshapeTest, testing::Combine(testing::Bool(), testing::Bool(), testing::Range(0, 10)));

TEST(NormalizeMoEOutputReshapeTest, ResolvesDynamicHiddenSizeAndTarget) {
    const auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, -1, 8});
    const auto indices = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1, 2});
    const auto weights = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{3, -1, 8});
    const auto gemm = std::make_shared<ov::op::internal::GatherMatmul>(input, weights, indices);
    const auto routing = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{2, -1, 1});
    const auto product = std::make_shared<ov::op::v1::Multiply>(gemm, routing);
    const auto reduction = std::make_shared<ov::op::v1::ReduceSum>(product, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0}), false);
    const auto target = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{2});
    const auto reshape = std::make_shared<ov::op::v1::Reshape>(reduction, target, true);
    const auto model = std::make_shared<ov::Model>(ov::OutputVector{reshape}, ov::ParameterVector{input, weights, indices, routing, target});
    ov::pass::Manager manager;
    manager.register_pass<ov::intel_gpu::NormalizeMoEOutputReshape>();
    manager.run_passes(model);
    EXPECT_FALSE(reshape->get_special_zero());
    const auto shape_model = std::make_shared<ov::Model>(ov::OutputVector{reshape->input_value(1)}, ov::ParameterVector{weights, indices, routing, target});
    for (const auto& node : shape_model->get_ordered_ops()) {
        EXPECT_FALSE(ov::is_type<ov::op::internal::GatherMatmul>(node));
    }
    for (const auto& [tokens, routing_tokens] : std::vector<std::pair<size_t, size_t>>{{4, 4}, {1, 4}, {4, 1}, {0, 1}, {1, 0}}) {
        for (const auto& values : std::vector<std::vector<int32_t>>{{0, 0}, {-1, 0}, {0, -1}, {2, 16}}) {
            ov::TensorVector inputs{ov::Tensor(ov::element::f16, {3, 8, 8}),
                                    ov::Tensor(ov::element::i32, {tokens, 2}),
                                    ov::Tensor(ov::element::f16, {2, routing_tokens, 1}),
                                    ov::Tensor(ov::element::i32, {2})};
            std::copy(values.begin(), values.end(), inputs.back().data<int32_t>());
            ov::TensorVector outputs{ov::Tensor(ov::element::i32, {2})};
            ASSERT_TRUE(shape_model->evaluate(outputs, inputs));
            const auto broadcast_tokens = static_cast<int32_t>(tokens == 1 ? routing_tokens : tokens);
            EXPECT_EQ(outputs[0].data<int32_t>()[0], values[0] == 0 ? broadcast_tokens : values[0]);
            EXPECT_EQ(outputs[0].data<int32_t>()[1], values[1] == 0 ? 8 : values[1]);
        }
    }
}

}  // namespace ov::test::intel_gpu
