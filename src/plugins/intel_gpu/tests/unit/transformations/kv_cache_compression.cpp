// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <optional>

#include "common_test_utils/ov_test_utils.hpp"

#include "openvino/core/model.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/variadic_split.hpp"

#include "intel_gpu/op/read_value.hpp"
#include "intel_gpu/op/read_values.hpp"
#include "intel_gpu/op/kv_cache.hpp"
#include "intel_gpu/op/kv_cache_compressed.hpp"
#include "intel_gpu/op/indirect_sdpa.hpp"
#include "intel_gpu/op/sparse_sdpa.hpp"

#include "plugin/transformations/kv_cache_compression.hpp"

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

constexpr int64_t sparse_head_size = 64;

struct SparseCacheModelOptions {
    bool key_shape_read = false;                  ///< a ShapeOf of the key cache is a model output
    std::optional<ov::PartialShape> initializer;  ///< the shape of a ReadValue initializer of both caches
};

ov::intel_gpu::op::SparseSDPA::Config sparse_sdpa_config() {
    ov::intel_gpu::op::SparseSDPA::Config config;
    config.num_heads = 4;
    config.num_kv_heads = 2;
    config.k_head_size = static_cast<size_t>(sparse_head_size);
    config.v_head_size = static_cast<size_t>(sparse_head_size);
    config.key_block_size = 16;
    return config;
}

// The GPU SparseSDPA op (see ConvertSparseAttentionToInternal) over the indirect key / value caches
// IndirectSparseAttentionOpt makes, with a Constant scale.
std::shared_ptr<ov::Model> make_sparse_attention_kv_cache_model(const SparseCacheModelOptions& options = {}) {
    const auto f16 = ov::element::f16;
    const auto query = std::make_shared<ov::op::v0::Parameter>(f16, ov::PartialShape{-1, 4, -1, sparse_head_size});
    const auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
    const auto indices = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1, 1, -1, 8});
    const auto count = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1, 1, -1});
    const auto key_length = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
    ov::ParameterVector params{query, beam_idx, indices, count, key_length};
    const auto cache = [&](const char* name) {
        const auto variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{-1, 2, -1, sparse_head_size}, f16, name});
        const auto current = std::make_shared<ov::op::v0::Parameter>(f16, ov::PartialShape{-1, 2, -1, sparse_head_size});
        params.push_back(current);
        std::shared_ptr<ov::intel_gpu::op::ReadValue> past;
        if (options.initializer) {
            const auto initializer = std::make_shared<ov::op::v0::Parameter>(f16, *options.initializer);
            params.push_back(initializer);
            past = std::make_shared<ov::intel_gpu::op::ReadValue>(initializer, variable);
        } else {
            past = std::make_shared<ov::intel_gpu::op::ReadValue>(variable);
        }
        const auto kv_cache = std::make_shared<ov::intel_gpu::op::KVCache>(past, current, beam_idx, variable, 2, 0, f16);
        kv_cache->get_rt_info()[ov::intel_gpu::op::KV_CACHE_REQUIRES_IDENTITY_BEAM_IDX] = true;
        return kv_cache;
    };
    const auto key = cache("k");
    const auto value = cache("v");
    const auto scale = ov::op::v0::Constant::create(f16, ov::Shape{}, {0.125f});
    const auto attention =
        std::make_shared<ov::intel_gpu::op::SparseSDPA>(ov::OutputVector{query, key->output(0), value->output(0), indices, count, key_length, scale},
                                                        sparse_sdpa_config());
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(attention)};
    if (options.key_shape_read) {
        results.push_back(std::make_shared<ov::op::v0::Result>(std::make_shared<ov::op::v3::ShapeOf>(key, ov::element::i32)));
    }
    return std::make_shared<ov::Model>(results, params);
}

std::shared_ptr<ov::Node> attention_of(const std::shared_ptr<ov::Model>& model) {
    return model->get_results()[0]->get_input_node_shared_ptr(0);
}

void compress(const std::shared_ptr<ov::Model>& model, ov::element::Type precision, bool supports_immad) {
    ov::pass::Manager manager;
    manager.register_pass<ov::intel_gpu::KVCacheCompression>(precision, supports_immad);
    manager.run_passes(model);
}

bool is_compressed(const std::shared_ptr<ov::Model>& model) {
    const auto sdpa = ov::as_type_ptr<ov::intel_gpu::op::SparseSDPA>(attention_of(model));
    return sdpa && sdpa->get_config().kv_compressed() &&
           ov::is_type<ov::intel_gpu::op::KVCacheCompressed>(sdpa->get_input_node_shared_ptr(ov::intel_gpu::op::SparseSDPA::InputIdx::KEY));
}

}  // namespace

// The sparse attention caches are quantized as the SDPA ones on systolic platforms (planar, per token, asymmetric) and
// the attention becomes a compressed SparseSDPA, which reads them.
TEST_F(TransformationTestsF, KVCacheCompressionSparseAttention) {
    model = make_sparse_attention_kv_cache_model();
    manager.register_pass<KVCacheCompression>(ov::element::i8, true);
    {
        const auto f16 = ov::element::f16;
        ov::op::internal::DynamicQuantize::Attributes dq_config;
        dq_config.quantization_type = ov::op::internal::DynamicQuantize::QuantizationType::Asymmetric;
        dq_config.quantization_dt = ov::element::i8;
        dq_config.scale_dt = f16;
        dq_config.zp_dt = ov::element::i8;
        dq_config.group_sizes = {1, 1, 1, UINT64_MAX};
        dq_config.scales_zp_output_order = {0, 1, 2, 3};
        dq_config.output_storage_type = ov::op::internal::DynamicQuantize::OutputStorageType::Planar;

        const auto query = std::make_shared<ov::op::v0::Parameter>(f16, ov::PartialShape{-1, 4, -1, sparse_head_size});
        const auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
        const auto indices = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1, 1, -1, 8});
        const auto count = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1, 1, -1});
        const auto key_length = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
        ov::ParameterVector params{query, beam_idx, indices, count, key_length};
        const auto cache = [&](const char* name) {
            // The cache variable holds the quantized data.
            const auto variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{{-1, 2, -1, sparse_head_size}, ov::element::i8, name});
            const auto current = std::make_shared<ov::op::v0::Parameter>(f16, ov::PartialShape{-1, 2, -1, sparse_head_size});
            params.push_back(current);
            const std::vector<ov::op::util::VariableInfo> infos{{{-1, 2, -1, sparse_head_size}, ov::element::i8, name},
                                                                {{-1, 2, -1, 1}, f16, name},
                                                                {{-1, 2, -1, 1}, ov::element::i8, name}};
            const auto past = std::make_shared<ov::intel_gpu::op::ReadValues>(variable, infos);
            const auto kv_cache =
                std::make_shared<ov::intel_gpu::op::KVCacheCompressed>(ov::OutputVector{past->output(0), current, beam_idx, past->output(1), past->output(2)},
                                                                       variable,
                                                                       false,
                                                                       2,
                                                                       0,
                                                                       dq_config);
            kv_cache->get_rt_info()[ov::intel_gpu::op::KV_CACHE_REQUIRES_IDENTITY_BEAM_IDX] = true;
            return kv_cache;
        };
        const auto key = cache("k");
        const auto value = cache("v");
        auto config = sparse_sdpa_config();
        config.key_quant.mode = ov::intel_gpu::op::KVQuant::Mode::BY_TOKEN;
        config.key_quant.bits = 8;
        config.value_quant = config.key_quant;
        const auto scale = ov::op::v0::Constant::create(f16, ov::Shape{}, {0.125f});
        const auto attention = std::make_shared<ov::intel_gpu::op::SparseSDPA>(ov::OutputVector{query,
                                                                                                key->output(0),
                                                                                                value->output(0),
                                                                                                indices,
                                                                                                count,
                                                                                                key_length,
                                                                                                key->output(2),
                                                                                                value->output(2),
                                                                                                key->output(3),
                                                                                                value->output(3),
                                                                                                scale},
                                                                               config);
        model_ref = std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(attention)}, params);
    }
    comparator.enable(FunctionsComparator::ATTRIBUTES);
}

// A shape read of an 8-bit cache moves to the compressed cache, so that the uncompressed KVCache drops out of the model.
TEST(KVCacheCompressionSparseAttention, ShapeReadOfCacheFollowsCompression) {
    SparseCacheModelOptions options;
    options.key_shape_read = true;
    const auto model = make_sparse_attention_kv_cache_model(options);
    compress(model, ov::element::i8, true);
    ASSERT_TRUE(is_compressed(model));
    const auto shape_read = model->get_results()[1]->get_input_node_shared_ptr(0);
    EXPECT_TRUE(ov::is_type<ov::intel_gpu::op::KVCacheCompressed>(shape_read->get_input_node_shared_ptr(0)));
    for (const auto& node : model->get_ordered_ops()) {
        EXPECT_TRUE(!ov::is_type<ov::intel_gpu::op::KVCache>(node) || ov::is_type<ov::intel_gpu::op::KVCacheCompressed>(node)) << node->get_friendly_name();
    }
}

// A 4-bit cache has f16 zero points; its empty initializer (as exported stateful models have) is quantized as it is.
TEST(KVCacheCompressionSparseAttention, Int4HasF16ZeroPoints) {
    SparseCacheModelOptions options;
    options.initializer = ov::PartialShape{-1, 2, 0, sparse_head_size};
    const auto model = make_sparse_attention_kv_cache_model(options);
    compress(model, ov::element::u4, true);
    ASSERT_TRUE(is_compressed(model));
    const auto sdpa = ov::as_type_ptr<ov::intel_gpu::op::SparseSDPA>(attention_of(model));
    EXPECT_EQ(sdpa->get_config().key_quant.bits, 4);
    EXPECT_EQ(sdpa->get_input_element_type(ov::intel_gpu::op::SparseSDPA::InputIdx::KEY_ZP), ov::element::f16);
}

// The kv_cache packs 4-bit codes, which neither a quantized initializer nor a shape read of the cache would see.
TEST(KVCacheCompressionSparseAttention, Int4KeepsUncompressedCachesWithInitializerOrShapeRead) {
    SparseCacheModelOptions with_initializer;
    with_initializer.initializer = ov::PartialShape{-1, 2, -1, sparse_head_size};
    auto model = make_sparse_attention_kv_cache_model(with_initializer);
    compress(model, ov::element::u4, true);
    EXPECT_FALSE(is_compressed(model));

    // The 8-bit caches have no packing.
    model = make_sparse_attention_kv_cache_model(with_initializer);
    compress(model, ov::element::i8, true);
    EXPECT_TRUE(is_compressed(model));

    SparseCacheModelOptions with_shape_read;
    with_shape_read.key_shape_read = true;
    model = make_sparse_attention_kv_cache_model(with_shape_read);
    compress(model, ov::element::u4, true);
    EXPECT_FALSE(is_compressed(model));
}

// SparseSDPA does not read the interleaved scales / zero points of non-systolic platforms, and f16 caches stay as they are.
TEST(KVCacheCompressionSparseAttention, KeepsUncompressedCachesWithoutImmadOrQuantizedPrecision) {
    auto model = make_sparse_attention_kv_cache_model();
    compress(model, ov::element::i8, false);
    EXPECT_FALSE(is_compressed(model));

    model = make_sparse_attention_kv_cache_model();
    compress(model, ov::element::f16, true);
    EXPECT_FALSE(is_compressed(model));
}

}  // namespace intel_gpu
}  // namespace test
}  // namespace ov
