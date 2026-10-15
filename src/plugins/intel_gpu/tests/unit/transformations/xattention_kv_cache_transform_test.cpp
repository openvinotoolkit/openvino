// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "intel_gpu/plugin/remote_context.hpp"
#include "intel_gpu/plugin/transformations_pipeline.hpp"
#include "intel_gpu/runtime/execution_config.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "test_utils.h"

#include "../../../src/plugin/transformations_pipeline.cpp"

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

}  // namespace

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

}  // namespace ov::test::intel_gpu
