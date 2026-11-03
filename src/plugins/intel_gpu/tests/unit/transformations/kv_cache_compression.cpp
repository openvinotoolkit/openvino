// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/transformations/kv_cache_compression.hpp"

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

        ov::ParameterVector params{ beam_idx, query, key_current, value_current };

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

        ov::OutputVector sdpa_inputs = { query, key_cache->output(0), value_cache->output(0) };

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

        ov::ResultVector results{ result };

        model = std::make_shared<ov::Model>(results, params);
        manager.register_pass<KVCacheCompression>(ov::element::i8, false);
    }
    {
        ov::op::internal::DynamicQuantize::Attributes dq_config;
        dq_config.quantization_type = ov::op::internal::DynamicQuantize::QuantizationType::Asymmetric;
        dq_config.quantization_dt = ov::element::i8;
        dq_config.scale_dt = ov::element::f16;
        dq_config.zp_dt = ov::element::f16;
        dq_config.group_sizes = { 1, 1, 1, UINT64_MAX };
        dq_config.scales_zp_output_order = { 0, 1, 2, 3 };
        dq_config.output_storage_type = ov::op::internal::DynamicQuantize::OutputStorageType::InterleavedScalesZP;

        auto query = std::make_shared<ov::op::v0::Parameter>(element_type, input_shape);
        auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1});

        auto key_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::f16, "v0"});
        auto key_current = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, input_shape);
        auto key_past_variable_infos = { ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::i8, "v0"},
                                         ov::op::util::VariableInfo{{1, 32, -1, 2}, ov::element::f16, "v0"} };
        auto key_past_compressed = std::make_shared<ov::intel_gpu::op::ReadValues>(key_variable, key_past_variable_infos);
        auto key_cache_inputs = ov::OutputVector{ key_past_compressed->output(0), key_current, beam_idx, key_past_compressed->output(1) };
        auto key_cache = std::make_shared<ov::intel_gpu::op::KVCacheCompressed>(key_cache_inputs,
                                                                                key_variable,
                                                                                false,
                                                                                concat_axis,
                                                                                gather_axis,
                                                                                dq_config);

        auto value_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::f16, "v1"});
        auto value_current = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, input_shape);
        auto value_past_variable_infos = { ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::i8, "v1"},
                                           ov::op::util::VariableInfo{{1, 32, -1, 2}, ov::element::f16, "v1"} };
        auto value_past_compressed = std::make_shared<ov::intel_gpu::op::ReadValues>(value_variable, value_past_variable_infos);
        auto value_cache_inputs = ov::OutputVector{ value_past_compressed->output(0), value_current, beam_idx, value_past_compressed->output(1) };
        auto value_cache = std::make_shared<ov::intel_gpu::op::KVCacheCompressed>(value_cache_inputs,
                                                                                  value_variable,
                                                                                  false,
                                                                                  concat_axis,
                                                                                  gather_axis,
                                                                                  dq_config);

        ov::ParameterVector params{ beam_idx, query, key_current, value_current };

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

        ov::OutputVector sdpa_inputs = { query, key_cache->output(0), value_cache->output(0) };
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

        ov::ResultVector results{ result };

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

        ov::ParameterVector params{ past_seq_len, beam_idx, query, key_current, value_current };

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

        ov::OutputVector sdpa_inputs = { query, key_cache->output(0), value_cache->output(0) };

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

        ov::ResultVector results{ result };

        model = std::make_shared<ov::Model>(results, params);
        manager.register_pass<KVCacheCompression>(ov::element::i8, false);
    }
    {
        ov::op::internal::DynamicQuantize::Attributes dq_config;
        dq_config.quantization_type = ov::op::internal::DynamicQuantize::QuantizationType::Asymmetric;
        dq_config.quantization_dt = ov::element::i8;
        dq_config.scale_dt = ov::element::f16;
        dq_config.zp_dt = ov::element::f16;
        dq_config.group_sizes = { 1, 1, 1, UINT64_MAX };
        dq_config.scales_zp_output_order = { 0, 1, 2, 3 };
        dq_config.output_storage_type = ov::op::internal::DynamicQuantize::OutputStorageType::InterleavedScalesZP;

        auto query = std::make_shared<ov::op::v0::Parameter>(element_type, input_shape);
        auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1});
        auto past_seq_len = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1});

        auto key_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::f16, "v0"});
        auto key_current = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, input_shape);
        auto key_past_variable_infos = { ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::i8, "v0"},
                                         ov::op::util::VariableInfo{{1, 32, -1, 2}, ov::element::f16, "v0"} };
        auto key_past_compressed = std::make_shared<ov::intel_gpu::op::ReadValues>(key_variable, key_past_variable_infos);
        auto key_cache_inputs = ov::OutputVector{ key_past_compressed->output(0), key_current, beam_idx, past_seq_len, key_past_compressed->output(1) };
        auto key_cache = std::make_shared<ov::intel_gpu::op::KVCacheCompressed>(key_cache_inputs,
                                                                                key_variable,
                                                                                true,
                                                                                concat_axis,
                                                                                gather_axis,
                                                                                dq_config);

        auto value_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::f16, "v1"});
        auto value_current = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, input_shape);
        auto value_past_variable_infos = { ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::i8, "v1"},
                                           ov::op::util::VariableInfo{{1, 32, -1, 2}, ov::element::f16, "v1"} };
        auto value_past_compressed = std::make_shared<ov::intel_gpu::op::ReadValues>(value_variable, value_past_variable_infos);
        auto value_cache_inputs = ov::OutputVector{value_past_compressed->output(0), value_current, beam_idx, past_seq_len, value_past_compressed->output(1)};
        auto value_cache = std::make_shared<ov::intel_gpu::op::KVCacheCompressed>(value_cache_inputs,
                                                                                  value_variable, 
                                                                                  true,
                                                                                  concat_axis,
                                                                                  gather_axis,
                                                                                  dq_config);

        ov::ParameterVector params{ past_seq_len, beam_idx, query, key_current, value_current };

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

        ov::OutputVector sdpa_inputs = { query, key_cache->output(0), value_cache->output(0) };
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

        ov::ResultVector results{ result };

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

        ov::ParameterVector params{ beam_idx, query, key_current, value_current, key_variable_initializer, value_variable_initializer };

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

        ov::OutputVector sdpa_inputs = { query, key_cache->output(0), value_cache->output(0) };

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

        ov::ResultVector results{ result };

        model = std::make_shared<ov::Model>(results, params);
        manager.register_pass<KVCacheCompression>(ov::element::i8, false);
    }
    {
        ov::op::internal::DynamicQuantize::Attributes dq_config;
        dq_config.quantization_type = ov::op::internal::DynamicQuantize::QuantizationType::Asymmetric;
        dq_config.quantization_dt = ov::element::i8;
        dq_config.scale_dt = ov::element::f16;
        dq_config.zp_dt = ov::element::f16;
        dq_config.group_sizes = { 1, 1, 1, UINT64_MAX };
        dq_config.scales_zp_output_order = { 0, 1, 2, 3 };
        dq_config.output_storage_type = ov::op::internal::DynamicQuantize::OutputStorageType::InterleavedScalesZP;

        auto query = std::make_shared<ov::op::v0::Parameter>(element_type, input_shape);
        auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1});

        auto key_past_variable_infos = { ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::i8, "v0"},
                                         ov::op::util::VariableInfo{{1, 32, -1, 2}, ov::element::f16, "v0"} };
        auto key_current = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, input_shape);
        auto key_variable_initializer = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, input_shape);
        auto key_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::f16, "v0"});

        auto key_initializer_dq =
            std::make_shared<ov::op::internal::DynamicQuantize>(key_variable_initializer, dq_config);
        auto key_past_initializers = ov::OutputVector{ key_initializer_dq->output(0), key_initializer_dq->output(1) };
        auto key_past_compressed = std::make_shared<ov::intel_gpu::op::ReadValues>(key_past_initializers, key_variable, key_past_variable_infos);
        auto key_cache_inputs = ov::OutputVector{ key_past_compressed->output(0), key_current, beam_idx, key_past_compressed->output(1) };
        auto key_cache = std::make_shared<ov::intel_gpu::op::KVCacheCompressed>(key_cache_inputs,
                                                                                key_variable,
                                                                                false,
                                                                                concat_axis,
                                                                                gather_axis,
                                                                                dq_config);

        auto value_past_variable_infos = { ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::i8, "v1"},
                                           ov::op::util::VariableInfo{{1, 32, -1, 2}, ov::element::f16, "v1"} };

        auto value_current = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, input_shape);
        auto value_variable_initializer = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, input_shape);
        auto value_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::f16, "v1"});

        auto value_initializer_dq =
            std::make_shared<ov::op::internal::DynamicQuantize>(value_variable_initializer, dq_config);
        auto value_past_initializers = ov::OutputVector{ value_initializer_dq->output(0), value_initializer_dq->output(1) };
        auto value_past_compressed = std::make_shared<ov::intel_gpu::op::ReadValues>(value_past_initializers, value_variable, value_past_variable_infos);
        auto value_cache_inputs = ov::OutputVector{ value_past_compressed->output(0), value_current, beam_idx, value_past_compressed->output(1) };
        auto value_cache = std::make_shared<ov::intel_gpu::op::KVCacheCompressed>(value_cache_inputs,
                                                                                  value_variable,
                                                                                  false,
                                                                                  concat_axis,
                                                                                  gather_axis,
                                                                                  dq_config);

        ov::ParameterVector params{ beam_idx, query, key_current, value_current, key_variable_initializer, value_variable_initializer };

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

        ov::OutputVector sdpa_inputs = { query, key_cache->output(0), value_cache->output(0) };
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

        ov::ResultVector results{ result };

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
        auto value_cache = std::make_shared<ov::intel_gpu::op::KVCache>(value_past, value_current_vs->output(2), beam_idx, value_variable, concat_axis, gather_axis);

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
        dq_config.group_sizes = { 1, 1, 1, UINT64_MAX };
        dq_config.scales_zp_output_order = { 0, 1, 2, 3 };
        dq_config.output_storage_type = ov::op::internal::DynamicQuantize::OutputStorageType::InterleavedScalesZP;

        auto query = std::make_shared<ov::op::v0::Parameter>(element::f16, input_shape);
        auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});

        auto key_past_variable_infos = { ov::op::util::VariableInfo{{-1, 32, -1, 80}, ov::element::i8, "v0"},
                                         ov::op::util::VariableInfo{{-1, 32, -1, 2}, ov::element::f16, "v0"} };

        auto key_current_data = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 32, -1, 240});
        auto key_vs_axis = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, 3);
        auto key_vs_split_length = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{3}, std::vector<int>{80, 80, -1});
        auto key_current_vs = std::make_shared<ov::op::v1::VariadicSplit>(key_current_data, key_vs_axis, key_vs_split_length);
        auto key_variable_initializer = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, input_shape);
        auto key_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{-1, 32, -1, 80}, ov::element::f16, "v0"});

        auto key_initializer_dq =
            std::make_shared<ov::op::internal::DynamicQuantize>(key_variable_initializer, dq_config);
        auto key_past_initializers = ov::OutputVector{key_initializer_dq->output(0), key_initializer_dq->output(1)};
        auto key_past_compressed = std::make_shared<ov::intel_gpu::op::ReadValues>(key_past_initializers, key_variable, key_past_variable_infos);
        auto key_cache_inputs = ov::OutputVector{key_past_compressed->output(0), key_current_vs->output(2), beam_idx, key_past_compressed->output(1)};
        auto key_cache = std::make_shared<ov::intel_gpu::op::KVCacheCompressed>(key_cache_inputs,
                                                                                key_variable,
                                                                                false,
                                                                                concat_axis,
                                                                                gather_axis,
                                                                                dq_config);

        auto value_past_variable_infos = {ov::op::util::VariableInfo{{-1, 32, -1, 80}, ov::element::i8, "v1"},
                                          ov::op::util::VariableInfo{{-1, 32, -1, 2}, ov::element::f16, "v1"}};

        auto value_current_data = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 32, -1, 240});
        auto value_vs_axis = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, 3);
        auto value_vs_split_length = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{3}, std::vector<int>{80, 80, -1});
        auto value_current_vs = std::make_shared<ov::op::v1::VariadicSplit>(value_current_data, value_vs_axis, value_vs_split_length);
        auto value_variable_initializer = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, input_shape);
        auto value_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::f16, "v1"});

        auto value_initializer_dq =
            std::make_shared<ov::op::internal::DynamicQuantize>(value_variable_initializer, dq_config);
        auto value_past_initializers = ov::OutputVector{value_initializer_dq->output(0), value_initializer_dq->output(1)};
        auto value_past_compressed = std::make_shared<ov::intel_gpu::op::ReadValues>(value_past_initializers, value_variable, value_past_variable_infos);
        auto value_cache_inputs = ov::OutputVector{value_past_compressed->output(0), value_current_vs->output(2), beam_idx, value_past_compressed->output(1)};
        auto value_cache = std::make_shared<ov::intel_gpu::op::KVCacheCompressed>(value_cache_inputs,
                                                                                  value_variable,
                                                                                  false,
                                                                                  concat_axis,
                                                                                  gather_axis,
                                                                                  dq_config);

        auto attn_mask = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 1, -1, -1});
        auto scale = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{}, 0.111816f);
        auto sdpa = std::make_shared<ov::intel_gpu::op::IndirectSDPA>(ov::OutputVector{query, key_cache->output(0), value_cache->output(0),
                                                                                       attn_mask, scale,
                                                                                       key_cache->output(2), value_cache->output(2)},
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

        ov::ParameterVector params{ beam_idx, query, key_current, value_current };

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

        ov::OutputVector sdpa_inputs = { query, key_cache->output(0), value_cache->output(0) };

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

        ov::ResultVector results{ result };

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
        dq_config.group_sizes = { 1, 1, 1, UINT64_MAX };
        dq_config.scales_zp_output_order = { 0, 1, 2, 3 };
        dq_config.output_storage_type = ov::op::internal::DynamicQuantize::OutputStorageType::Planar;

        auto query = std::make_shared<ov::op::v0::Parameter>(element_type, input_shape);
        auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1});

        // For Planar mode, ReadValues produces 3 outputs: data(i8), scale(f16), zp(i8)
        auto key_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::f16, "v0"});
        auto key_current = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, input_shape);
        auto key_past_variable_infos = { ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::i8, "v0"},
                                         ov::op::util::VariableInfo{{1, 32, -1, 1}, ov::element::f16, "v0"},
                                         ov::op::util::VariableInfo{{1, 32, -1, 1}, ov::element::i8, "v0"} };
        auto key_past_compressed = std::make_shared<ov::intel_gpu::op::ReadValues>(key_variable, key_past_variable_infos);
        auto key_cache_inputs = ov::OutputVector{ key_past_compressed->output(0), key_current, beam_idx,
                                                  key_past_compressed->output(1), key_past_compressed->output(2) };
        auto key_cache = std::make_shared<ov::intel_gpu::op::KVCacheCompressed>(key_cache_inputs,
                                                                                key_variable,
                                                                                false,
                                                                                concat_axis,
                                                                                gather_axis,
                                                                                dq_config);

        auto value_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::f16, "v1"});
        auto value_current = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, input_shape);
        auto value_past_variable_infos = { ov::op::util::VariableInfo{{1, 32, -1, 80}, ov::element::i8, "v1"},
                                           ov::op::util::VariableInfo{{1, 32, -1, 1}, ov::element::f16, "v1"},
                                           ov::op::util::VariableInfo{{1, 32, -1, 1}, ov::element::i8, "v1"} };
        auto value_past_compressed = std::make_shared<ov::intel_gpu::op::ReadValues>(value_variable, value_past_variable_infos);
        auto value_cache_inputs = ov::OutputVector{ value_past_compressed->output(0), value_current, beam_idx,
                                                    value_past_compressed->output(1), value_past_compressed->output(2) };
        auto value_cache = std::make_shared<ov::intel_gpu::op::KVCacheCompressed>(value_cache_inputs,
                                                                                  value_variable,
                                                                                  false,
                                                                                  concat_axis,
                                                                                  gather_axis,
                                                                                  dq_config);

        ov::ParameterVector params{ beam_idx, query, key_current, value_current };

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

        ov::OutputVector sdpa_inputs = { query, key_cache->output(0), value_cache->output(0) };
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

        ov::ResultVector results{ result };

        model_ref = std::make_shared<ov::Model>(results, params);
    }
}

namespace {

// SparseSDPA over two plain KVCache nodes, optionally behind a beam_idx Gather, with a static batch of 1 unless `dynamic_batch`.
std::shared_ptr<ov::Model> make_sparse_sdpa_model(bool with_gather, bool with_scale, bool dynamic_batch = false) {
    using In = ov::intel_gpu::op::SparseSDPA::InputIdx;
    const ov::Dimension batch = dynamic_batch ? ov::Dimension::dynamic() : ov::Dimension(1);
    const ov::PartialShape cache_shape{batch, 2, -1, 64};
    auto query = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{batch, 4, -1, 64});
    auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{batch});
    ov::ParameterVector params{query, beam_idx};

    auto make_cache = [&](const std::string& name) {
        auto variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{cache_shape, ov::element::f16, name});
        auto current = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, cache_shape);
        params.push_back(current);
        auto past = std::make_shared<ov::intel_gpu::op::ReadValue>(variable);
        ov::Output<ov::Node> past_out = past->output(0);
        if (with_gather) {
            past_out = std::make_shared<ov::op::v8::Gather>(past, beam_idx, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0}));
        }
        return std::make_shared<ov::intel_gpu::op::KVCache>(past_out, current, variable, 2, ov::element::f16);
    };
    auto key_cache = make_cache("k");
    auto value_cache = make_cache("v");

    auto selected = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{batch, 1, -1, 8});
    auto count = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{batch, 1, -1});
    auto key_length = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{batch});
    params.push_back(selected);
    params.push_back(count);
    params.push_back(key_length);

    ov::OutputVector args(In::COUNT_MIN);
    args[In::QUERY] = query;
    args[In::KEY] = key_cache->output(0);
    args[In::VALUE] = value_cache->output(0);
    args[In::SELECTED_BLOCK_INDICES] = selected;
    args[In::SELECTED_BLOCK_COUNT] = count;
    args[In::KEY_LENGTH] = key_length;
    if (with_scale) {
        args.push_back(ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {0.125f}));
    }
    ov::intel_gpu::op::SparseSDPA::Config config;
    config.num_heads = 4;
    config.num_kv_heads = 2;
    config.k_head_size = 64;
    config.v_head_size = 64;
    config.key_block_size = 4;
    auto sdpa = std::make_shared<ov::intel_gpu::op::SparseSDPA>(args, config);
    return std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(sdpa)}, params);
}

std::shared_ptr<ov::intel_gpu::op::SparseSDPA> find_sparse_sdpa(const std::shared_ptr<ov::Model>& model) {
    for (const auto& op : model->get_ordered_ops()) {
        if (auto sdpa = ov::as_type_ptr<ov::intel_gpu::op::SparseSDPA>(op)) {
            return sdpa;
        }
    }
    return nullptr;
}

void run_kv_cache_compression(const std::shared_ptr<ov::Model>& model, ov::element::Type dt, bool supports_immad) {
    ov::pass::Manager manager;
    manager.register_pass<KVCacheCompression>(dt, supports_immad);
    manager.run_passes(model);
}

}  // namespace

TEST(KVCacheCompressionSparseSDPA, CompressesGatheredCaches) {
    using In = ov::intel_gpu::op::SparseSDPA::InputIdx;
    for (bool with_scale : {false, true}) {
        auto model = make_sparse_sdpa_model(true, with_scale, true);
        run_kv_cache_compression(model, ov::element::i8, true);
        auto sdpa = find_sparse_sdpa(model);
        ASSERT_NE(sdpa, nullptr);
        ASSERT_TRUE(sdpa->get_config().kv_compressed);
        ASSERT_EQ(sdpa->get_input_size(), In::COUNT_COMPRESSED);
        EXPECT_EQ(sdpa->get_input_element_type(In::KEY), ov::element::i8);
        EXPECT_EQ(sdpa->get_input_element_type(In::VALUE), ov::element::i8);
        EXPECT_EQ(sdpa->get_input_element_type(In::KEY_SCALE), ov::element::f16);
        EXPECT_EQ(sdpa->get_input_element_type(In::KEY_ZP), ov::element::i8);
        EXPECT_TRUE(ov::is_type<ov::op::v0::Constant>(sdpa->get_input_node_shared_ptr(In::SCALE)));
        for (size_t data : {In::KEY, In::VALUE}) {
            auto cache = ov::as_type_ptr<ov::intel_gpu::op::KVCacheCompressed>(sdpa->get_input_node_shared_ptr(data));
            ASSERT_NE(cache, nullptr);
            EXPECT_TRUE(cache->get_indirect());
            // The Gather is replaced by the beam table machinery: beam_idx feeds the cache directly.
            EXPECT_TRUE(ov::is_type<ov::op::v0::Parameter>(cache->get_input_node_shared_ptr(2)));
            EXPECT_TRUE(ov::is_type<ov::intel_gpu::op::ReadValues>(cache->get_input_node_shared_ptr(0)));
        }
        EXPECT_EQ(sdpa->get_input_node_shared_ptr(In::KEY_SCALE), sdpa->get_input_node_shared_ptr(In::KEY));
        EXPECT_EQ(sdpa->input_value(In::KEY_SCALE).get_index(), 2u);
        EXPECT_EQ(sdpa->input_value(In::KEY_ZP).get_index(), 3u);
        EXPECT_EQ(sdpa->input_value(In::VALUE_ZP).get_index(), 3u);
    }
}

TEST(KVCacheCompressionSparseSDPA, CompressesSingleSequenceCachesWithoutGather) {
    using In = ov::intel_gpu::op::SparseSDPA::InputIdx;
    auto model = make_sparse_sdpa_model(false, true);
    run_kv_cache_compression(model, ov::element::u8, true);
    auto sdpa = find_sparse_sdpa(model);
    ASSERT_NE(sdpa, nullptr);
    ASSERT_TRUE(sdpa->get_config().kv_compressed);
    auto cache = ov::as_type_ptr<ov::intel_gpu::op::KVCacheCompressed>(sdpa->get_input_node_shared_ptr(In::KEY));
    ASSERT_NE(cache, nullptr);
    // No Gather: a constant identity beam index of the single sequence.
    EXPECT_TRUE(ov::is_type<ov::op::v0::Constant>(cache->get_input_node_shared_ptr(2)));
}

TEST(KVCacheCompressionSparseSDPA, KeepsF16CachesWhenCompressionIsNotApplicable) {
    auto expect_f16 = [](const std::shared_ptr<ov::Model>& model) {
        auto sdpa = find_sparse_sdpa(model);
        ASSERT_NE(sdpa, nullptr);
        EXPECT_FALSE(sdpa->get_config().kv_compressed);
        EXPECT_EQ(sdpa->get_input_element_type(ov::intel_gpu::op::SparseSDPA::InputIdx::KEY), ov::element::f16);
    };
    {  // 4-bit precision: the kernels read 8-bit data only
        auto model = make_sparse_sdpa_model(true, true);
        run_kv_cache_compression(model, ov::element::u4, true);
        expect_f16(model);
    }
    {  // no systolic platform: planar i8 zero points are an immad layout
        auto model = make_sparse_sdpa_model(true, true);
        run_kv_cache_compression(model, ov::element::i8, false);
        expect_f16(model);
    }
    {  // several sequences without a gather cannot be expressed by a constant beam index
        auto model = make_sparse_sdpa_model(false, true, true);
        run_kv_cache_compression(model, ov::element::i8, true);
        expect_f16(model);
    }
}

}  // namespace intel_gpu
}  // namespace test
}  // namespace ov
