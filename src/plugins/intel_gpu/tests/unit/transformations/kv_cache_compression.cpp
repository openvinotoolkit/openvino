// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/transformations/kv_cache_compression.hpp"

#include <optional>
#include <utility>

#include "common_test_utils/ov_test_utils.hpp"
#include "intel_gpu/op/indirect_sdpa.hpp"
#include "intel_gpu/op/kv_cache.hpp"
#include "intel_gpu/op/kv_cache_compressed.hpp"
#include "intel_gpu/op/read_value.hpp"
#include "intel_gpu/op/read_values.hpp"
#include "intel_gpu/op/sparse_sdpa.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/manager.hpp"

using namespace testing;
using namespace ov::intel_gpu;

namespace ov {
namespace test {
namespace intel_gpu {

TEST_F(TransformationTestsF, KVCacheCompression) {
    bool causal = false;
    bool with_mask = true;
    bool with_scale = true;
    size_t concat_axis = 2;
    size_t gather_axis = 0;
    ov::element::Type_t element_type = ov::element::f16;
    std::vector<int64_t> qkv_order = {0, 1, 2, 3};
    std::shared_ptr<ov::Node> mask = nullptr;
    std::shared_ptr<ov::Node> scale = nullptr;
    ov::PartialShape input_shape = ov::PartialShape{1, 32, -1, 80};

    {
        auto query = std::make_shared<ov::op::v0::Parameter>(element_type, input_shape);
        auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1});

        auto key_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::f16, "v0"});
        auto key_current = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, input_shape);
        auto key_past = std::make_shared<ov::intel_gpu::op::ReadValue>(key_variable);
        auto key_cache = std::make_shared<ov::intel_gpu::op::KVCache>(key_past, key_current, beam_idx, key_variable, concat_axis, gather_axis);

        auto value_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::f16, "v1"});
        auto value_current = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, input_shape);
        auto value_past = std::make_shared<ov::intel_gpu::op::ReadValue>(value_variable);
        auto value_cache = std::make_shared<ov::intel_gpu::op::KVCache>(value_past, value_current, beam_idx, value_variable, concat_axis, gather_axis);

        ov::ParameterVector params{beam_idx, query, key_current, value_current};

        if (with_mask) {
            auto attn_mask = std::make_shared<ov::op::v0::Parameter>(element_type, ov::PartialShape::dynamic(4));
            mask = attn_mask;
            params.push_back(attn_mask);
        }

        if (with_mask && with_scale) {
            auto scale_input = std::make_shared<ov::op::v0::Parameter>(element_type, ov::PartialShape{1});
            scale = scale_input;
            params.push_back(scale_input);
        }

        ov::OutputVector sdpa_inputs = {query, key_cache->output(0), value_cache->output(0)};

        if (mask) {
            sdpa_inputs.push_back(mask);
        }

        if (scale) {
            sdpa_inputs.push_back(scale);
        }

        std::shared_ptr<ov::Node> sdpa = nullptr;
        sdpa = std::make_shared<ov::intel_gpu::op::IndirectSDPA>(sdpa_inputs,
                                                                 key_cache->output(1),
                                                                 causal,
                                                                 gather_axis,
                                                                 qkv_order,
                                                                 qkv_order,
                                                                 qkv_order,
                                                                 ov::intel_gpu::op::SDPA::default_order(4));

        auto result = std::make_shared<ov::op::v0::Result>(sdpa);

        ov::ResultVector results{result};

        model = std::make_shared<ov::Model>(results, params);
        manager.register_pass<KVCacheCompression>(ov::element::i8, false);
    }
    {
        ov::op::internal::DynamicQuantize::Attributes dq_config;
        dq_config.quantization_type = ov::op::internal::DynamicQuantize::QuantizationType::Asymmetric;
        dq_config.quantization_dt = ov::element::i8;
        dq_config.scale_dt = ov::element::f16;
        dq_config.zp_dt = ov::element::f16;
        dq_config.group_sizes = {1, 1, 1, UINT64_MAX};
        dq_config.scales_zp_output_order = {0, 1, 2, 3};
        dq_config.output_storage_type = ov::op::internal::DynamicQuantize::OutputStorageType::InterleavedScalesZP;

        auto query = std::make_shared<ov::op::v0::Parameter>(element_type, input_shape);
        auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1});

        auto key_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::f16, "v0"});
        auto key_current = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, input_shape);
        auto key_past_variable_infos = {ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::i8, "v0"},
                                        ov::op::util::VariableInfo{{1, 32, -1, 2}, ov::element::f16, "v0"}};
        auto key_past_compressed = std::make_shared<ov::intel_gpu::op::ReadValues>(key_variable, key_past_variable_infos);
        auto key_cache_inputs = ov::OutputVector{key_past_compressed->output(0), key_current, beam_idx, key_past_compressed->output(1)};
        auto key_cache = std::make_shared<ov::intel_gpu::op::KVCacheCompressed>(key_cache_inputs, key_variable, false, concat_axis, gather_axis, dq_config);

        auto value_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::f16, "v1"});
        auto value_current = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, input_shape);
        auto value_past_variable_infos = {ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::i8, "v1"},
                                          ov::op::util::VariableInfo{{1, 32, -1, 2}, ov::element::f16, "v1"}};
        auto value_past_compressed = std::make_shared<ov::intel_gpu::op::ReadValues>(value_variable, value_past_variable_infos);
        auto value_cache_inputs = ov::OutputVector{value_past_compressed->output(0), value_current, beam_idx, value_past_compressed->output(1)};
        auto value_cache =
            std::make_shared<ov::intel_gpu::op::KVCacheCompressed>(value_cache_inputs, value_variable, false, concat_axis, gather_axis, dq_config);

        ov::ParameterVector params{beam_idx, query, key_current, value_current};

        if (with_mask) {
            auto attn_input = std::make_shared<ov::op::v0::Parameter>(element_type, ov::PartialShape::dynamic(4));
            mask = attn_input;
            params.push_back(attn_input);
        }

        if (with_mask && with_scale) {
            auto scale_input = std::make_shared<ov::op::v0::Parameter>(element_type, ov::PartialShape{1});
            scale = scale_input;
            params.push_back(scale_input);
        }

        ov::OutputVector sdpa_inputs = {query, key_cache->output(0), value_cache->output(0)};
        if (mask) {
            sdpa_inputs.push_back(mask);
        }

        if (scale) {
            sdpa_inputs.push_back(scale);
        }

        sdpa_inputs.push_back(key_cache->output(2));
        sdpa_inputs.push_back(value_cache->output(2));

        std::shared_ptr<ov::Node> sdpa = nullptr;
        sdpa = std::make_shared<ov::intel_gpu::op::IndirectSDPA>(sdpa_inputs,
                                                                 key_cache->output(1),
                                                                 causal,
                                                                 gather_axis,
                                                                 qkv_order,
                                                                 qkv_order,
                                                                 qkv_order,
                                                                 ov::intel_gpu::op::SDPA::default_order(4),
                                                                 dq_config);

        auto result = std::make_shared<ov::op::v0::Result>(sdpa);

        ov::ResultVector results{result};

        model_ref = std::make_shared<ov::Model>(results, params);
    }
}

TEST_F(TransformationTestsF, KVCacheCompressionWithPastSeqLen) {
    bool causal = false;
    bool with_mask = true;
    bool with_scale = true;
    size_t concat_axis = 2;
    size_t gather_axis = 0;
    ov::element::Type_t element_type = ov::element::f16;
    std::vector<int64_t> qkv_order = {0, 1, 2, 3};
    std::shared_ptr<ov::Node> mask = nullptr;
    std::shared_ptr<ov::Node> scale = nullptr;
    ov::PartialShape input_shape = ov::PartialShape{1, 32, -1, 80};

    {
        auto query = std::make_shared<ov::op::v0::Parameter>(element_type, input_shape);
        auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1});
        auto past_seq_len = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1});

        auto key_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::f16, "v0"});
        auto key_current = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, input_shape);
        auto key_past = std::make_shared<ov::intel_gpu::op::ReadValue>(key_variable);
        auto key_cache = std::make_shared<ov::intel_gpu::op::KVCache>(key_past, key_current, beam_idx, past_seq_len, key_variable, concat_axis, gather_axis);

        auto value_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::f16, "v1"});
        auto value_current = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, input_shape);
        auto value_past = std::make_shared<ov::intel_gpu::op::ReadValue>(value_variable);
        auto value_cache =
            std::make_shared<ov::intel_gpu::op::KVCache>(value_past, value_current, beam_idx, past_seq_len, value_variable, concat_axis, gather_axis);

        ov::ParameterVector params{past_seq_len, beam_idx, query, key_current, value_current};

        if (with_mask) {
            auto attn_mask = std::make_shared<ov::op::v0::Parameter>(element_type, ov::PartialShape::dynamic(4));
            mask = attn_mask;
            params.push_back(attn_mask);
        }

        if (with_mask && with_scale) {
            auto scale_input = std::make_shared<ov::op::v0::Parameter>(element_type, ov::PartialShape{1});
            scale = scale_input;
            params.push_back(scale_input);
        }

        ov::OutputVector sdpa_inputs = {query, key_cache->output(0), value_cache->output(0)};

        if (mask) {
            sdpa_inputs.push_back(mask);
        }

        if (scale) {
            sdpa_inputs.push_back(scale);
        }

        std::shared_ptr<ov::Node> sdpa = nullptr;
        sdpa = std::make_shared<ov::intel_gpu::op::IndirectSDPA>(sdpa_inputs,
                                                                 key_cache->output(1),
                                                                 causal,
                                                                 gather_axis,
                                                                 qkv_order,
                                                                 qkv_order,
                                                                 qkv_order,
                                                                 ov::intel_gpu::op::SDPA::default_order(4));

        auto result = std::make_shared<ov::op::v0::Result>(sdpa);

        ov::ResultVector results{result};

        model = std::make_shared<ov::Model>(results, params);
        manager.register_pass<KVCacheCompression>(ov::element::i8, false);
    }
    {
        ov::op::internal::DynamicQuantize::Attributes dq_config;
        dq_config.quantization_type = ov::op::internal::DynamicQuantize::QuantizationType::Asymmetric;
        dq_config.quantization_dt = ov::element::i8;
        dq_config.scale_dt = ov::element::f16;
        dq_config.zp_dt = ov::element::f16;
        dq_config.group_sizes = {1, 1, 1, UINT64_MAX};
        dq_config.scales_zp_output_order = {0, 1, 2, 3};
        dq_config.output_storage_type = ov::op::internal::DynamicQuantize::OutputStorageType::InterleavedScalesZP;

        auto query = std::make_shared<ov::op::v0::Parameter>(element_type, input_shape);
        auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1});
        auto past_seq_len = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1});

        auto key_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::f16, "v0"});
        auto key_current = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, input_shape);
        auto key_past_variable_infos = {ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::i8, "v0"},
                                        ov::op::util::VariableInfo{{1, 32, -1, 2}, ov::element::f16, "v0"}};
        auto key_past_compressed = std::make_shared<ov::intel_gpu::op::ReadValues>(key_variable, key_past_variable_infos);
        auto key_cache_inputs = ov::OutputVector{key_past_compressed->output(0), key_current, beam_idx, past_seq_len, key_past_compressed->output(1)};
        auto key_cache = std::make_shared<ov::intel_gpu::op::KVCacheCompressed>(key_cache_inputs, key_variable, true, concat_axis, gather_axis, dq_config);

        auto value_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::f16, "v1"});
        auto value_current = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, input_shape);
        auto value_past_variable_infos = {ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::i8, "v1"},
                                          ov::op::util::VariableInfo{{1, 32, -1, 2}, ov::element::f16, "v1"}};
        auto value_past_compressed = std::make_shared<ov::intel_gpu::op::ReadValues>(value_variable, value_past_variable_infos);
        auto value_cache_inputs = ov::OutputVector{value_past_compressed->output(0), value_current, beam_idx, past_seq_len, value_past_compressed->output(1)};
        auto value_cache =
            std::make_shared<ov::intel_gpu::op::KVCacheCompressed>(value_cache_inputs, value_variable, true, concat_axis, gather_axis, dq_config);

        ov::ParameterVector params{past_seq_len, beam_idx, query, key_current, value_current};

        if (with_mask) {
            auto attn_input = std::make_shared<ov::op::v0::Parameter>(element_type, ov::PartialShape::dynamic(4));
            mask = attn_input;
            params.push_back(attn_input);
        }

        if (with_mask && with_scale) {
            auto scale_input = std::make_shared<ov::op::v0::Parameter>(element_type, ov::PartialShape{1});
            scale = scale_input;
            params.push_back(scale_input);
        }

        ov::OutputVector sdpa_inputs = {query, key_cache->output(0), value_cache->output(0)};
        if (mask) {
            sdpa_inputs.push_back(mask);
        }

        if (scale) {
            sdpa_inputs.push_back(scale);
        }

        sdpa_inputs.push_back(key_cache->output(2));
        sdpa_inputs.push_back(value_cache->output(2));

        std::shared_ptr<ov::Node> sdpa = nullptr;
        sdpa = std::make_shared<ov::intel_gpu::op::IndirectSDPA>(sdpa_inputs,
                                                                 key_cache->output(1),
                                                                 causal,
                                                                 gather_axis,
                                                                 qkv_order,
                                                                 qkv_order,
                                                                 qkv_order,
                                                                 ov::intel_gpu::op::SDPA::default_order(4),
                                                                 dq_config);

        auto result = std::make_shared<ov::op::v0::Result>(sdpa);

        ov::ResultVector results{result};

        model_ref = std::make_shared<ov::Model>(results, params);
    }
}

TEST_F(TransformationTestsF, KVCacheCompressionWithInitializers) {
    bool causal = false;
    bool with_mask = true;
    bool with_scale = true;
    size_t concat_axis = 2;
    size_t gather_axis = 0;
    ov::element::Type_t element_type = ov::element::f16;
    std::vector<int64_t> qkv_order = {0, 1, 2, 3};
    std::shared_ptr<ov::Node> mask = nullptr;
    std::shared_ptr<ov::Node> scale = nullptr;
    ov::PartialShape input_shape = ov::PartialShape{1, 32, -1, 80};

    {
        auto query = std::make_shared<ov::op::v0::Parameter>(element_type, input_shape);
        auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1});

        auto key_variable_initializer = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, input_shape);
        auto key_current = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, input_shape);
        auto key_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::f16, "v0"});
        auto key_past = std::make_shared<ov::intel_gpu::op::ReadValue>(key_variable_initializer, key_variable);
        auto key_cache = std::make_shared<ov::intel_gpu::op::KVCache>(key_past, key_current, beam_idx, key_variable, concat_axis, gather_axis);

        auto value_variable_initializer = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, input_shape);
        auto value_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::f16, "v1"});
        auto value_current = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, input_shape);
        auto value_past = std::make_shared<ov::intel_gpu::op::ReadValue>(value_variable_initializer, value_variable);
        auto value_cache = std::make_shared<ov::intel_gpu::op::KVCache>(value_past, value_current, beam_idx, value_variable, concat_axis, gather_axis);

        ov::ParameterVector params{beam_idx, query, key_current, value_current, key_variable_initializer, value_variable_initializer};

        if (with_mask) {
            auto attn_mask = std::make_shared<ov::op::v0::Parameter>(element_type, ov::PartialShape::dynamic(4));
            mask = attn_mask;
            params.push_back(attn_mask);
        }

        if (with_mask && with_scale) {
            auto scale_input = std::make_shared<ov::op::v0::Parameter>(element_type, ov::PartialShape{1});
            scale = scale_input;
            params.push_back(scale_input);
        }

        ov::OutputVector sdpa_inputs = {query, key_cache->output(0), value_cache->output(0)};

        if (mask) {
            sdpa_inputs.push_back(mask);
        }

        if (scale) {
            sdpa_inputs.push_back(scale);
        }

        std::shared_ptr<ov::Node> sdpa = nullptr;
        sdpa = std::make_shared<ov::intel_gpu::op::IndirectSDPA>(sdpa_inputs,
                                                                 key_cache->output(1),
                                                                 causal,
                                                                 gather_axis,
                                                                 qkv_order,
                                                                 qkv_order,
                                                                 qkv_order,
                                                                 ov::intel_gpu::op::SDPA::default_order(4));

        auto result = std::make_shared<ov::op::v0::Result>(sdpa);

        ov::ResultVector results{result};

        model = std::make_shared<ov::Model>(results, params);
        manager.register_pass<KVCacheCompression>(ov::element::i8, false);
    }
    {
        ov::op::internal::DynamicQuantize::Attributes dq_config;
        dq_config.quantization_type = ov::op::internal::DynamicQuantize::QuantizationType::Asymmetric;
        dq_config.quantization_dt = ov::element::i8;
        dq_config.scale_dt = ov::element::f16;
        dq_config.zp_dt = ov::element::f16;
        dq_config.group_sizes = {1, 1, 1, UINT64_MAX};
        dq_config.scales_zp_output_order = {0, 1, 2, 3};
        dq_config.output_storage_type = ov::op::internal::DynamicQuantize::OutputStorageType::InterleavedScalesZP;

        auto query = std::make_shared<ov::op::v0::Parameter>(element_type, input_shape);
        auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1});

        auto key_past_variable_infos = {ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::i8, "v0"},
                                        ov::op::util::VariableInfo{{1, 32, -1, 2}, ov::element::f16, "v0"}};
        auto key_current = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, input_shape);
        auto key_variable_initializer = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, input_shape);
        auto key_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::f16, "v0"});

        auto key_initializer_dq = std::make_shared<ov::op::internal::DynamicQuantize>(key_variable_initializer, dq_config);
        auto key_past_initializers = ov::OutputVector{key_initializer_dq->output(0), key_initializer_dq->output(1)};
        auto key_past_compressed = std::make_shared<ov::intel_gpu::op::ReadValues>(key_past_initializers, key_variable, key_past_variable_infos);
        auto key_cache_inputs = ov::OutputVector{key_past_compressed->output(0), key_current, beam_idx, key_past_compressed->output(1)};
        auto key_cache = std::make_shared<ov::intel_gpu::op::KVCacheCompressed>(key_cache_inputs, key_variable, false, concat_axis, gather_axis, dq_config);

        auto value_past_variable_infos = {ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::i8, "v1"},
                                          ov::op::util::VariableInfo{{1, 32, -1, 2}, ov::element::f16, "v1"}};

        auto value_current = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, input_shape);
        auto value_variable_initializer = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, input_shape);
        auto value_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::f16, "v1"});

        auto value_initializer_dq = std::make_shared<ov::op::internal::DynamicQuantize>(value_variable_initializer, dq_config);
        auto value_past_initializers = ov::OutputVector{value_initializer_dq->output(0), value_initializer_dq->output(1)};
        auto value_past_compressed = std::make_shared<ov::intel_gpu::op::ReadValues>(value_past_initializers, value_variable, value_past_variable_infos);
        auto value_cache_inputs = ov::OutputVector{value_past_compressed->output(0), value_current, beam_idx, value_past_compressed->output(1)};
        auto value_cache =
            std::make_shared<ov::intel_gpu::op::KVCacheCompressed>(value_cache_inputs, value_variable, false, concat_axis, gather_axis, dq_config);

        ov::ParameterVector params{beam_idx, query, key_current, value_current, key_variable_initializer, value_variable_initializer};

        if (with_mask) {
            auto attn_input = std::make_shared<ov::op::v0::Parameter>(element_type, ov::PartialShape::dynamic(4));
            mask = attn_input;
            params.push_back(attn_input);
        }

        if (with_mask && with_scale) {
            auto scale_input = std::make_shared<ov::op::v0::Parameter>(element_type, ov::PartialShape{1});
            scale = scale_input;
            params.push_back(scale_input);
        }

        ov::OutputVector sdpa_inputs = {query, key_cache->output(0), value_cache->output(0)};
        if (mask) {
            sdpa_inputs.push_back(mask);
        }

        if (scale) {
            sdpa_inputs.push_back(scale);
        }

        sdpa_inputs.push_back(key_cache->output(2));
        sdpa_inputs.push_back(value_cache->output(2));

        std::shared_ptr<ov::Node> sdpa = nullptr;
        sdpa = std::make_shared<ov::intel_gpu::op::IndirectSDPA>(sdpa_inputs,
                                                                 key_cache->output(1),
                                                                 causal,
                                                                 gather_axis,
                                                                 qkv_order,
                                                                 qkv_order,
                                                                 qkv_order,
                                                                 ov::intel_gpu::op::SDPA::default_order(4),
                                                                 dq_config);

        auto result = std::make_shared<ov::op::v0::Result>(sdpa);

        ov::ResultVector results{result};

        model_ref = std::make_shared<ov::Model>(results, params);
    }
}

TEST_F(TransformationTestsF, NewTestKVCacheCompression) {
    ov::PartialShape input_shape = ov::PartialShape{-1, 32, -1, 80};
    std::vector<int64_t> qkv_order = {0, 1, 2, 3};
    bool causal = false;
    size_t gather_axis = 0;
    size_t concat_axis = -2;
    {
        auto query = std::make_shared<ov::op::v0::Parameter>(element::f16, input_shape);
        auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});

        auto key_current_data = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 32, -1, 240});
        auto key_vs_axis = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, 3);
        auto key_vs_split_length = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{3}, std::vector<int>{80, 80, -1});
        auto key_current_vs = std::make_shared<ov::op::v1::VariadicSplit>(key_current_data, key_vs_axis, key_vs_split_length);
        auto key_variable_initializer = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, input_shape);
        auto key_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{-1, 32, -1, 80}, ov::element::f16, "v0"});
        auto key_past = std::make_shared<ov::intel_gpu::op::ReadValue>(key_variable_initializer, key_variable);
        auto key_cache = std::make_shared<ov::intel_gpu::op::KVCache>(key_past, key_current_vs->output(2), beam_idx, key_variable, concat_axis, gather_axis);

        auto value_current_data = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 32, -1, 240});
        auto value_vs_axis = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, 3);
        auto value_vs_split_length = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{3}, std::vector<int>{80, 80, -1});
        auto value_current_vs = std::make_shared<ov::op::v1::VariadicSplit>(value_current_data, value_vs_axis, value_vs_split_length);
        auto value_variable_initializer = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, input_shape);
        auto value_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{-1, 32, -1, 80}, ov::element::f16, "v1"});
        auto value_past = std::make_shared<ov::intel_gpu::op::ReadValue>(value_variable_initializer, value_variable);
        auto value_cache =
            std::make_shared<ov::intel_gpu::op::KVCache>(value_past, value_current_vs->output(2), beam_idx, value_variable, concat_axis, gather_axis);

        auto attn_mask = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 1, -1, -1});
        auto scale = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{}, 0.111816f);
        auto sdpa = std::make_shared<ov::intel_gpu::op::IndirectSDPA>(ov::OutputVector{query, key_cache->output(0), value_cache->output(0), attn_mask, scale},
                                                                      key_cache->output(1),
                                                                      causal,
                                                                      gather_axis,
                                                                      qkv_order,
                                                                      qkv_order,
                                                                      qkv_order,
                                                                      ov::intel_gpu::op::SDPA::default_order(4));
        auto result = std::make_shared<ov::op::v0::Result>(sdpa);

        auto params = ParameterVector{query, beam_idx, key_variable_initializer, key_current_data, value_variable_initializer, value_current_data, attn_mask};
        model = std::make_shared<ov::Model>(ResultVector{result}, params);
        manager.register_pass<KVCacheCompression>(ov::element::i8, false);
    }
    {
        ov::op::internal::DynamicQuantize::Attributes dq_config;
        dq_config.quantization_type = ov::op::internal::DynamicQuantize::QuantizationType::Asymmetric;
        dq_config.quantization_dt = ov::element::i8;
        dq_config.scale_dt = ov::element::f16;
        dq_config.zp_dt = ov::element::f16;
        dq_config.group_sizes = {1, 1, 1, UINT64_MAX};
        dq_config.scales_zp_output_order = {0, 1, 2, 3};
        dq_config.output_storage_type = ov::op::internal::DynamicQuantize::OutputStorageType::InterleavedScalesZP;

        auto query = std::make_shared<ov::op::v0::Parameter>(element::f16, input_shape);
        auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});

        auto key_past_variable_infos = {ov::op::util::VariableInfo{{-1, 32, -1, 80}, ov::element::i8, "v0"},
                                        ov::op::util::VariableInfo{{-1, 32, -1, 2}, ov::element::f16, "v0"}};

        auto key_current_data = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 32, -1, 240});
        auto key_vs_axis = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, 3);
        auto key_vs_split_length = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{3}, std::vector<int>{80, 80, -1});
        auto key_current_vs = std::make_shared<ov::op::v1::VariadicSplit>(key_current_data, key_vs_axis, key_vs_split_length);
        auto key_variable_initializer = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, input_shape);
        auto key_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{-1, 32, -1, 80}, ov::element::f16, "v0"});

        auto key_initializer_dq = std::make_shared<ov::op::internal::DynamicQuantize>(key_variable_initializer, dq_config);
        auto key_past_initializers = ov::OutputVector{key_initializer_dq->output(0), key_initializer_dq->output(1)};
        auto key_past_compressed = std::make_shared<ov::intel_gpu::op::ReadValues>(key_past_initializers, key_variable, key_past_variable_infos);
        auto key_cache_inputs = ov::OutputVector{key_past_compressed->output(0), key_current_vs->output(2), beam_idx, key_past_compressed->output(1)};
        auto key_cache = std::make_shared<ov::intel_gpu::op::KVCacheCompressed>(key_cache_inputs, key_variable, false, concat_axis, gather_axis, dq_config);

        auto value_past_variable_infos = {ov::op::util::VariableInfo{{-1, 32, -1, 80}, ov::element::i8, "v1"},
                                          ov::op::util::VariableInfo{{-1, 32, -1, 2}, ov::element::f16, "v1"}};

        auto value_current_data = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 32, -1, 240});
        auto value_vs_axis = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, 3);
        auto value_vs_split_length = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{3}, std::vector<int>{80, 80, -1});
        auto value_current_vs = std::make_shared<ov::op::v1::VariadicSplit>(value_current_data, value_vs_axis, value_vs_split_length);
        auto value_variable_initializer = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, input_shape);
        auto value_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::f16, "v1"});

        auto value_initializer_dq = std::make_shared<ov::op::internal::DynamicQuantize>(value_variable_initializer, dq_config);
        auto value_past_initializers = ov::OutputVector{value_initializer_dq->output(0), value_initializer_dq->output(1)};
        auto value_past_compressed = std::make_shared<ov::intel_gpu::op::ReadValues>(value_past_initializers, value_variable, value_past_variable_infos);
        auto value_cache_inputs = ov::OutputVector{value_past_compressed->output(0), value_current_vs->output(2), beam_idx, value_past_compressed->output(1)};
        auto value_cache =
            std::make_shared<ov::intel_gpu::op::KVCacheCompressed>(value_cache_inputs, value_variable, false, concat_axis, gather_axis, dq_config);

        auto attn_mask = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 1, -1, -1});
        auto scale = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{}, 0.111816f);
        auto sdpa = std::make_shared<ov::intel_gpu::op::IndirectSDPA>(
            ov::OutputVector{query, key_cache->output(0), value_cache->output(0), attn_mask, scale, key_cache->output(2), value_cache->output(2)},
            key_cache->output(1),
            causal,
            gather_axis,
            qkv_order,
            qkv_order,
            qkv_order,
            ov::intel_gpu::op::SDPA::default_order(4),
            dq_config);
        auto result = std::make_shared<ov::op::v0::Result>(sdpa);

        auto params = ParameterVector{query, beam_idx, key_variable_initializer, key_current_data, value_variable_initializer, value_current_data, attn_mask};
        model_ref = std::make_shared<ov::Model>(ResultVector{result}, params);
    }
}

// Test KV cache compression with Planar output storage (supports_immad=true),
// where scales and zero points are stored as separate buffers.
// This exercises the code path fixed for separate key_zp/value_zp kernel arguments in sdpa_opt.cl.
TEST_F(TransformationTestsF, KVCacheCompressionPlanar) {
    bool causal = false;
    bool with_mask = true;
    bool with_scale = true;
    size_t concat_axis = 2;
    size_t gather_axis = 0;
    ov::element::Type_t element_type = ov::element::f16;
    std::vector<int64_t> qkv_order = {0, 1, 2, 3};
    std::shared_ptr<ov::Node> mask = nullptr;
    std::shared_ptr<ov::Node> scale = nullptr;
    ov::PartialShape input_shape = ov::PartialShape{1, 32, -1, 80};

    {
        auto query = std::make_shared<ov::op::v0::Parameter>(element_type, input_shape);
        auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1});

        auto key_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::f16, "v0"});
        auto key_current = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, input_shape);
        auto key_past = std::make_shared<ov::intel_gpu::op::ReadValue>(key_variable);
        auto key_cache = std::make_shared<ov::intel_gpu::op::KVCache>(key_past, key_current, beam_idx, key_variable, concat_axis, gather_axis);

        auto value_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::f16, "v1"});
        auto value_current = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, input_shape);
        auto value_past = std::make_shared<ov::intel_gpu::op::ReadValue>(value_variable);
        auto value_cache = std::make_shared<ov::intel_gpu::op::KVCache>(value_past, value_current, beam_idx, value_variable, concat_axis, gather_axis);

        ov::ParameterVector params{beam_idx, query, key_current, value_current};

        if (with_mask) {
            auto attn_mask = std::make_shared<ov::op::v0::Parameter>(element_type, ov::PartialShape::dynamic(4));
            mask = attn_mask;
            params.push_back(attn_mask);
        }

        if (with_mask && with_scale) {
            auto scale_input = std::make_shared<ov::op::v0::Parameter>(element_type, ov::PartialShape{1});
            scale = scale_input;
            params.push_back(scale_input);
        }

        ov::OutputVector sdpa_inputs = {query, key_cache->output(0), value_cache->output(0)};

        if (mask) {
            sdpa_inputs.push_back(mask);
        }

        if (scale) {
            sdpa_inputs.push_back(scale);
        }

        std::shared_ptr<ov::Node> sdpa = nullptr;
        sdpa = std::make_shared<ov::intel_gpu::op::IndirectSDPA>(sdpa_inputs,
                                                                 key_cache->output(1),
                                                                 causal,
                                                                 gather_axis,
                                                                 qkv_order,
                                                                 qkv_order,
                                                                 qkv_order,
                                                                 ov::intel_gpu::op::SDPA::default_order(4));

        auto result = std::make_shared<ov::op::v0::Result>(sdpa);

        ov::ResultVector results{result};

        model = std::make_shared<ov::Model>(results, params);
        // supports_immad=true triggers Planar output storage with separate ZP buffers
        manager.register_pass<KVCacheCompression>(ov::element::i8, true);
    }
    {
        ov::op::internal::DynamicQuantize::Attributes dq_config;
        dq_config.quantization_type = ov::op::internal::DynamicQuantize::QuantizationType::Asymmetric;
        dq_config.quantization_dt = ov::element::i8;
        dq_config.scale_dt = ov::element::f16;
        dq_config.zp_dt = ov::element::i8;
        dq_config.group_sizes = {1, 1, 1, UINT64_MAX};
        dq_config.scales_zp_output_order = {0, 1, 2, 3};
        dq_config.output_storage_type = ov::op::internal::DynamicQuantize::OutputStorageType::Planar;

        auto query = std::make_shared<ov::op::v0::Parameter>(element_type, input_shape);
        auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1});

        // For Planar mode, ReadValues produces 3 outputs: data(i8), scale(f16), zp(i8)
        auto key_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::f16, "v0"});
        auto key_current = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, input_shape);
        auto key_past_variable_infos = {ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::i8, "v0"},
                                        ov::op::util::VariableInfo{{1, 32, -1, 1}, ov::element::f16, "v0"},
                                        ov::op::util::VariableInfo{{1, 32, -1, 1}, ov::element::i8, "v0"}};
        auto key_past_compressed = std::make_shared<ov::intel_gpu::op::ReadValues>(key_variable, key_past_variable_infos);
        auto key_cache_inputs =
            ov::OutputVector{key_past_compressed->output(0), key_current, beam_idx, key_past_compressed->output(1), key_past_compressed->output(2)};
        auto key_cache = std::make_shared<ov::intel_gpu::op::KVCacheCompressed>(key_cache_inputs, key_variable, false, concat_axis, gather_axis, dq_config);

        auto value_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::f16, "v1"});
        auto value_current = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, input_shape);
        auto value_past_variable_infos = {ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::i8, "v1"},
                                          ov::op::util::VariableInfo{{1, 32, -1, 1}, ov::element::f16, "v1"},
                                          ov::op::util::VariableInfo{{1, 32, -1, 1}, ov::element::i8, "v1"}};
        auto value_past_compressed = std::make_shared<ov::intel_gpu::op::ReadValues>(value_variable, value_past_variable_infos);
        auto value_cache_inputs =
            ov::OutputVector{value_past_compressed->output(0), value_current, beam_idx, value_past_compressed->output(1), value_past_compressed->output(2)};
        auto value_cache =
            std::make_shared<ov::intel_gpu::op::KVCacheCompressed>(value_cache_inputs, value_variable, false, concat_axis, gather_axis, dq_config);

        ov::ParameterVector params{beam_idx, query, key_current, value_current};

        if (with_mask) {
            auto attn_input = std::make_shared<ov::op::v0::Parameter>(element_type, ov::PartialShape::dynamic(4));
            mask = attn_input;
            params.push_back(attn_input);
        }

        if (with_mask && with_scale) {
            auto scale_input = std::make_shared<ov::op::v0::Parameter>(element_type, ov::PartialShape{1});
            scale = scale_input;
            params.push_back(scale_input);
        }

        ov::OutputVector sdpa_inputs = {query, key_cache->output(0), value_cache->output(0)};
        if (mask) {
            sdpa_inputs.push_back(mask);
        }

        if (scale) {
            sdpa_inputs.push_back(scale);
        }

        // For Planar mode: key_scale, value_scale, key_zp, value_zp
        sdpa_inputs.push_back(key_cache->output(2));
        sdpa_inputs.push_back(value_cache->output(2));
        sdpa_inputs.push_back(key_cache->output(3));
        sdpa_inputs.push_back(value_cache->output(3));

        std::shared_ptr<ov::Node> sdpa = nullptr;
        sdpa = std::make_shared<ov::intel_gpu::op::IndirectSDPA>(sdpa_inputs,
                                                                 key_cache->output(1),
                                                                 causal,
                                                                 gather_axis,
                                                                 qkv_order,
                                                                 qkv_order,
                                                                 qkv_order,
                                                                 ov::intel_gpu::op::SDPA::default_order(4),
                                                                 dq_config);

        auto result = std::make_shared<ov::op::v0::Result>(sdpa);

        ov::ResultVector results{result};

        model_ref = std::make_shared<ov::Model>(results, params);
    }
}

namespace {

using ov::intel_gpu::op::KVQuant;
using Mode = KVQuant::Mode;

enum class ExtraConsumer { None, KeyCache, ValueCache, KeyPast };

// SparseSDPA over two plain KVCache nodes, optionally behind a beam_idx Gather, with a static batch of 1 unless `dynamic_batch`.
struct SparseSdpaSpec {
    bool with_gather = true;
    bool with_scale = true;
    bool dynamic_batch = false;
    ExtraConsumer extra_consumer = ExtraConsumer::None;
};

ov::op::internal::DynamicQuantize::Attributes planar_cache_attributes(const KVQuant& quant) {
    using DQ = ov::op::internal::DynamicQuantize;
    DQ::Attributes attrs;
    attrs.quantization_type = quant.symmetric ? DQ::QuantizationType::Symmetric : DQ::QuantizationType::Asymmetric;
    attrs.output_storage_type = DQ::OutputStorageType::Planar;
    attrs.group_sizes = {1, 1, 1, UINT64_MAX};
    attrs.quantization_dt = ov::element::i8;
    attrs.scale_dt = ov::element::f16;
    attrs.zp_dt = quant.bits == 4 ? ov::element::f16 : ov::element::i8;
    attrs.scales_zp_output_order = {0, 1, 2, 3};
    return attrs;
}

// Original model without `quant`; the expected compressed model with it (the KVCache pair is replaced by KVCacheCompressed).
std::shared_ptr<ov::Model> make_sparse_sdpa_model(const SparseSdpaSpec& spec, const std::optional<std::pair<KVQuant, KVQuant>>& quant = std::nullopt) {
    using In = ov::intel_gpu::op::SparseSDPA::InputIdx;
    const ov::Dimension batch = spec.dynamic_batch ? ov::Dimension::dynamic() : ov::Dimension(1);
    const ov::PartialShape cache_shape{batch, 2, -1, 64};
    const ov::PartialShape param_shape{batch, 2, -1, 1};
    auto query = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{batch, 4, -1, 64});
    auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{batch});
    ov::ParameterVector params{query, beam_idx};
    ov::ResultVector results;

    auto make_cache = [&](const std::string& name, const KVQuant* cache_quant, bool extra_past_consumer) -> std::shared_ptr<ov::Node> {
        // The pass retypes the Variable to the compressed data type.
        auto variable =
            std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{cache_shape, cache_quant ? ov::element::i8 : ov::element::f16, name});
        auto current = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, cache_shape);
        params.push_back(current);
        if (!cache_quant) {
            auto past = std::make_shared<ov::intel_gpu::op::ReadValue>(variable);
            if (extra_past_consumer) {
                results.push_back(std::make_shared<ov::op::v0::Result>(past));
            }
            ov::Output<ov::Node> past_out = past->output(0);
            if (spec.with_gather) {
                past_out = std::make_shared<ov::op::v8::Gather>(past, beam_idx, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0}));
            }
            return std::make_shared<ov::intel_gpu::op::KVCache>(past_out, current, variable, 2, ov::element::f16);
        }
        const auto attrs = planar_cache_attributes(*cache_quant);
        std::vector<ov::op::util::VariableInfo> infos{ov::op::util::VariableInfo{cache_shape, ov::element::i8, name},
                                                      ov::op::util::VariableInfo{param_shape, ov::element::f16, name}};
        if (!cache_quant->symmetric) {
            infos.push_back(ov::op::util::VariableInfo{param_shape, attrs.zp_dt, name});
        }
        auto past = std::make_shared<ov::intel_gpu::op::ReadValues>(variable, infos);
        ov::Output<ov::Node> beam = beam_idx->output(0);
        if (!spec.with_gather) {
            // A single sequence: the constant identity beam index.
            beam = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, std::vector<int32_t>{0});
        }
        ov::OutputVector inputs{past->output(0), current, beam, past->output(1)};
        if (!cache_quant->symmetric) {
            inputs.push_back(past->output(2));
        }
        return std::make_shared<ov::intel_gpu::op::KVCacheCompressed>(inputs, variable, false, 2, 0, attrs);
    };
    auto key_cache = make_cache("k", quant ? &quant->first : nullptr, spec.extra_consumer == ExtraConsumer::KeyPast);
    auto value_cache = make_cache("v", quant ? &quant->second : nullptr, false);
    if (spec.extra_consumer == ExtraConsumer::KeyCache) {
        results.push_back(std::make_shared<ov::op::v0::Result>(key_cache->output(0)));
    }
    if (spec.extra_consumer == ExtraConsumer::ValueCache) {
        results.push_back(std::make_shared<ov::op::v0::Result>(value_cache->output(0)));
    }

    auto selected = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{batch, 1, -1, 8});
    auto count = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{batch, 1, -1});
    auto key_length = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{batch});
    params.push_back(selected);
    params.push_back(count);
    params.push_back(key_length);

    ov::intel_gpu::op::SparseSDPA::Config config;
    config.num_heads = 4;
    config.num_kv_heads = 2;
    config.k_head_size = 64;
    config.v_head_size = 64;
    config.key_block_size = 4;
    if (quant) {
        config.key_quant = quant->first;
        config.value_quant = quant->second;
    }
    ov::OutputVector args(config.scale_input_idx());
    args[In::QUERY] = query;
    args[In::KEY] = key_cache->output(0);
    args[In::VALUE] = value_cache->output(0);
    args[In::SELECTED_BLOCK_INDICES] = selected;
    args[In::SELECTED_BLOCK_COUNT] = count;
    args[In::KEY_LENGTH] = key_length;
    if (quant) {
        args[In::KEY_SCALE] = key_cache->output(2);
        args[In::VALUE_SCALE] = value_cache->output(2);
        if (!quant->first.symmetric) {
            args[In::KEY_ZP] = key_cache->output(3);
            args[In::VALUE_ZP] = value_cache->output(3);
        }
    }
    if (spec.with_scale) {
        args.push_back(ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {0.125f}));
    }
    auto sdpa = std::make_shared<ov::intel_gpu::op::SparseSDPA>(args, config);
    results.insert(results.begin(), std::make_shared<ov::op::v0::Result>(sdpa));
    return std::make_shared<ov::Model>(results, params);
}

struct SparseSdpaCompressionParams {
    std::string name;
    SparseSdpaSpec spec;
    ov::element::Type dt;
    KVQuant key, value;                    // requested by the QSA configuration
    KVQuant expected_key, expected_value;  // the quantization of the compressed caches
};

SparseSdpaSpec make_spec(bool with_gather, bool with_scale, bool dynamic_batch) {
    SparseSdpaSpec spec;
    spec.with_gather = with_gather;
    spec.with_scale = with_scale;
    spec.dynamic_batch = dynamic_batch;
    return spec;
}

SparseSdpaCompressionParams make_params(const std::string& name,
                                        const SparseSdpaSpec& spec,
                                        ov::element::Type dt,
                                        const KVQuant& key = {},
                                        const KVQuant& value = {}) {
    // The dense caches are quantized per token: a BY_CHANNEL key request falls back to BY_TOKEN; without a QSA request the
    // caches follow the global KV cache precision.
    const KVQuant by_precision{Mode::BY_TOKEN, static_cast<uint8_t>(dt.bitwidth()), false, 0};
    KVQuant expected_key = key.enabled() ? KVQuant{Mode::BY_TOKEN, key.bits, key.symmetric, 0} : by_precision;
    KVQuant expected_value = value.enabled() ? value : by_precision;
    return {name, spec, dt, key, value, expected_key, expected_value};
}

}  // namespace

class KVCacheCompressionSparseSDPA : public TransformationTestsF, public testing::WithParamInterface<SparseSdpaCompressionParams> {
public:
    KVCacheCompressionSparseSDPA() {
        comparator.enable(FunctionsComparator::ATTRIBUTES);
        comparator.enable(FunctionsComparator::CONST_VALUES);
    }

    static std::string get_test_case_name(const testing::TestParamInfo<SparseSdpaCompressionParams>& info) {
        return info.param.name;
    }

protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        const auto& p = GetParam();
        model = make_sparse_sdpa_model(p.spec);
        model_ref = make_sparse_sdpa_model(p.spec, std::make_pair(p.expected_key, p.expected_value));
        manager.register_pass<KVCacheCompression>(p.dt, true, p.key, p.value);
    }
};

TEST_P(KVCacheCompressionSparseSDPA, CompressesCaches) {}

INSTANTIATE_TEST_SUITE_P(
    KVCacheCompression,
    KVCacheCompressionSparseSDPA,
    testing::Values(
        // Gathered caches follow the KV cache precision.
        make_params("gather_i8", make_spec(true, false, true), ov::element::i8),
        make_params("gather_i8_scale", make_spec(true, true, true), ov::element::i8),
        make_params("gather_u8_scale", make_spec(true, true, true), ov::element::u8),
        make_params("gather_u4_scale", make_spec(true, true, true), ov::element::u4),
        // A single sequence gets the constant identity beam index.
        make_params("single_sequence_u8_scale", make_spec(false, true, false), ov::element::u8),
        // The QSA quantization decides the caches.
        make_params("qsa_i8_token", make_spec(true, true, true), ov::element::i8, {Mode::BY_TOKEN, 8, false, 0}, {Mode::BY_TOKEN, 8, false, 0}),
        make_params("qsa_i8_channel_key", make_spec(true, true, true), ov::element::i8, {Mode::BY_CHANNEL, 8, false, 16}, {Mode::BY_TOKEN, 8, false, 0}),
        make_params("qsa_u4_channel_key", make_spec(true, true, true), ov::element::u4, {Mode::BY_CHANNEL, 4, false, 32}, {Mode::BY_TOKEN, 4, false, 0}),
        make_params("qsa_u4_token", make_spec(true, true, true), ov::element::u4, {Mode::BY_TOKEN, 4, false, 0}, {Mode::BY_TOKEN, 4, false, 0}),
        make_params("qsa_i8_channel_key_symmetric",
                    make_spec(true, true, true),
                    ov::element::i8,
                    {Mode::BY_CHANNEL, 8, true, 16},
                    {Mode::BY_TOKEN, 8, true, 0}),
        make_params("qsa_u4_token_symmetric", make_spec(true, true, true), ov::element::u4, {Mode::BY_TOKEN, 4, true, 0}, {Mode::BY_TOKEN, 4, true, 0})),
    KVCacheCompressionSparseSDPA::get_test_case_name);

// f16 precision: the caches stay uncompressed
TEST_F(TransformationTestsF, KVCacheCompressionSparseSDPA_F16PrecisionKeepsCaches) {
    model = make_sparse_sdpa_model(make_spec(true, true, false));
    manager.register_pass<KVCacheCompression>(ov::element::f16, true);
}

// no systolic platform: planar i8 zero points are an immad layout
TEST_F(TransformationTestsF, KVCacheCompressionSparseSDPA_NoImmadKeepsCaches) {
    model = make_sparse_sdpa_model(make_spec(true, true, false));
    manager.register_pass<KVCacheCompression>(ov::element::i8, false);
}

// several sequences without a gather cannot be expressed by a constant beam index
TEST_F(TransformationTestsF, KVCacheCompressionSparseSDPA_DynamicBatchWithoutGatherKeepsCaches) {
    model = make_sparse_sdpa_model(make_spec(false, true, true));
    manager.register_pass<KVCacheCompression>(ov::element::i8, true);
}

// the caches follow the global precision: an 8-bit precision cannot hold a 4-bit quantization
TEST_F(TransformationTestsF, KVCacheCompressionSparseSDPA_PrecisionMismatchKeepsCaches) {
    model = make_sparse_sdpa_model(make_spec(true, true, false));
    manager.register_pass<KVCacheCompression>(ov::element::i8, true, KVQuant{Mode::BY_TOKEN, 4, false, 0}, KVQuant{Mode::BY_TOKEN, 4, false, 0});
}

// a cache read elsewhere would see the compressed state of the shared Variable
TEST_F(TransformationTestsF, KVCacheCompressionSparseSDPA_KeyCacheWithExtraConsumerKeepsCaches) {
    auto spec = make_spec(true, true, false);
    spec.extra_consumer = ExtraConsumer::KeyCache;
    model = make_sparse_sdpa_model(spec);
    manager.register_pass<KVCacheCompression>(ov::element::i8, true);
}

TEST_F(TransformationTestsF, KVCacheCompressionSparseSDPA_ValueCacheWithExtraConsumerKeepsCaches) {
    auto spec = make_spec(false, true, false);
    spec.extra_consumer = ExtraConsumer::ValueCache;
    model = make_sparse_sdpa_model(spec);
    manager.register_pass<KVCacheCompression>(ov::element::i8, true);
}

TEST_F(TransformationTestsF, KVCacheCompressionSparseSDPA_PastStateWithExtraConsumerKeepsCaches) {
    auto spec = make_spec(true, true, false);
    spec.extra_consumer = ExtraConsumer::KeyPast;
    model = make_sparse_sdpa_model(spec);
    manager.register_pass<KVCacheCompression>(ov::element::i8, true);
}

}  // namespace intel_gpu
}  // namespace test
}  // namespace ov
