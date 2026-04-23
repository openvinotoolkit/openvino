// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "serialization.hpp"

#include "attention.hpp"
#include "host_flash_attention.hpp"
#include "intel_npu/config/config.hpp"
#include "intel_npu/config/npuw.hpp"
#include "lazy_tensor.hpp"
#include "logging.hpp"
#include "moe_transformations/moe_transformation.hpp"
#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/reference/convert.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "openvino/util/mmap_object.hpp"
#include "pyramid_attention.hpp"
#include "spatial.hpp"
#include "util.hpp"

namespace {

std::streamsize checked_stream_size(const std::size_t size) {
    if (size > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
        OPENVINO_THROW("Blob size is too large to be represented on a std::streamsize!");
    }
    return static_cast<std::streamsize>(size);
}

}  // namespace

// NOTE: This construtor should only be used when exporting blobs
ov::npuw::s11n::WeightsContext::WeightsContext(bool _is_weightless,
                                               const std::unordered_map<const void*, std::size_t>& _const_to_offset)
    : is_weightless(_is_weightless),
      const_to_offset(_const_to_offset) {}

// NOTE: This construtor can and should only be used when importing blobs
ov::npuw::s11n::WeightsContext::WeightsContext(const ov::npuw::s11n::WeightsPtr& _weights,
                                               const std::string& _weights_path,
                                               const s11n::WeightsContext::ConstsCache& _consts_cache,
                                               const BF16Cache& _bf16_consts,
                                               const ov::FileHandleProvider& _handle_provider)
    : weights(_weights),
      weights_path(_weights_path),
      consts_cache(_consts_cache),
      bf16_consts(_bf16_consts),
      handle_provider(_handle_provider) {
    is_weightless = _weights || !_consts_cache.empty();
}

ov::npuw::s11n::BF16Cache ov::npuw::s11n::get_bf16_consts(const std::shared_ptr<ov::Model>& model) {
    ov::npuw::s11n::BF16Cache bf16_cache;
    for (auto&& node_ptr : model->get_ordered_ops()) {
        if (const auto c = ov::as_type_ptr<ov::op::v0::Constant>(node_ptr)) {
            if (c->get_element_type() != ov::element::bf16) {
                continue;
            }
            auto rt_info = c->get_rt_info();
            auto weightless_cache_attr = rt_info.find(ov::WeightlessCacheAttribute::get_type_info_static());
            if (weightless_cache_attr == rt_info.end()) {
                continue;
            }
            std::size_t offset = weightless_cache_attr->second.as<ov::WeightlessCacheAttribute>().bin_offset;
            bf16_cache.insert({offset, c->get_byte_size()});
        }
    }
    return bf16_cache;
}

void ov::npuw::s11n::serialize(Stream& stream, std::streampos& var) {
    stream.bytes(&var, sizeof var);
}

void ov::npuw::s11n::serialize(Stream& stream, std::string& var) {
    if (stream.output()) {
        auto var_size = var.size();
        stream.bytes(&var_size, sizeof var_size);
        if (!var.empty()) {
            stream.bytes(var.data(), checked_stream_size(var.size()));
        }
    } else {
        std::size_t var_size = 0;
        stream.bytes(&var_size, sizeof var_size);
        var.resize(var_size);
        if (var_size != 0) {
            stream.bytes(var.data(), checked_stream_size(var_size));
        }
    }
}

void ov::npuw::s11n::serialize(Stream& stream, bool& var) {
    stream.bytes(&var, sizeof var);
}

void ov::npuw::s11n::serialize(Stream& stream, float& var) {
    stream.bytes(&var, sizeof var);
}

void ov::npuw::s11n::serialize(Stream& stream, ov::npuw::compiled::Spatial& var) {
    stream & var.params & var.range & var.nway & var.out_dim & var.nway_iters & var.tail_size;
}

void ov::npuw::s11n::serialize(Stream& stream, ov::npuw::compiled::Spatial::Param& var) {
    stream & var.idx & var.dim;
}

void ov::npuw::s11n::serialize(Stream& stream, ov::npuw::compiled::Attention& var) {
    stream & var.query_size & var.context_size & var.params & var.mask_idx & var.attend_all;
}

void ov::npuw::s11n::serialize(Stream& stream, ov::npuw::compiled::Attention::Param& var) {
    stream & var.idx & var.dim;
}

void ov::npuw::s11n::serialize(Stream& stream, ov::npuw::compiled::PyramidAttention& var) {
    stream & var.query_size & var.full_context_size & var._context_lengths & var._attention_infos;
}

void ov::npuw::s11n::serialize(Stream& stream, ov::npuw::compiled::PyramidAttentionInfo& var) {
    stream & var.params & var.mask_idx & var.query_size & var.context_length;
}

void ov::npuw::s11n::serialize(Stream& stream, ov::npuw::compiled::PyramidAttentionInfo::Param& var) {
    stream & var.idx & var.dim;
}

void ov::npuw::s11n::serialize(Stream& stream, ov::npuw::compiled::HostFlashAttention& var) {
    auto& info = var._sdpa_attention_info;
    stream & info._query_size & info._context_size & info._k_seq_dim & info._v_seq_dim & info._sdpa_indices.query &
        info._sdpa_indices.past_key & info._sdpa_indices.past_value & info._sdpa_indices.present_key &
        info._sdpa_indices.present_value & info._sdpa_indices.attention_mask & info._tile_input_indices.q &
        info._tile_input_indices.k & info._tile_input_indices.v & info._tile_input_indices.mask &
        info._tile_input_indices.acc & info._tile_input_indices.max & info._tile_input_indices.d &
        info._tile_output_indices.acc & info._tile_output_indices.max & info._tile_output_indices.d & var._tile_size &
        var._can_use_tensor_view;
}

void ov::npuw::s11n::serialize(Stream& stream, ov::npuw::compiled::MoEExperts& var) {
    stream & var.num_experts & var.expert_hidden_dim & var.num_active_experts & var.input_token_count &
        var._router_scores_idx & var._expert_input_param_idx & var._param_mapping;
}

void ov::npuw::s11n::serialize(Stream& stream, ov::npuw::compiled::MoEDownstream& var) {
    stream & var.total_experts_num & var.active_experts_num & var.expert_output_param_idx;
}

void ov::npuw::s11n::serialize(Stream& stream, ov::Tensor& var) {
    transfer_tensor(stream, var);
}

void ov::npuw::s11n::serialize(Stream& stream, ::intel_npu::Config& var) {
    std::string str;
    if (stream.output()) {
        str = var.toString();
    }
    stream & str;
    if (stream.input()) {
        var.fromString(str);
    }
}

void ov::npuw::s11n::serialize(Stream& stream, ov::Output<const ov::Node>& var) {
    if (stream.output()) {
        auto elem_type = var.get_element_type().to_string();
        auto shape = var.get_partial_shape().to_string();
        auto names = var.get_names();
        stream & elem_type & shape & names;
    } else {
        OPENVINO_THROW("ov::Output<const ov::Node> is write-only in NPUW serialization");
    }
}

void ov::npuw::s11n::transfer_tensor(Stream& stream, ov::Tensor& var, const TensorAllocator& allocator) {
    if (stream.output()) {
        bool is_initialized = static_cast<bool>(var);
        stream & is_initialized;
        if (!is_initialized) {
            return;
        }

        auto type_str = var.get_element_type().to_string();
        auto shape = var.get_shape();
        auto byte_size = var.get_byte_size();
        stream & type_str & shape & byte_size;

        ov::Tensor tensor = var;
        if (!var.is_continuous()) {
            tensor = ov::Tensor(var.get_element_type(), var.get_shape());
            var.copy_to(tensor);
        }
        NPUW_ASSERT(tensor);
        stream.bytes(tensor.data(), tensor.get_byte_size());
        return;
    }

    bool is_initialized = false;
    stream & is_initialized;
    if (!is_initialized) {
        var = ov::Tensor();
        return;
    }

    std::string type_str;
    stream & type_str;
    ov::element::Type type(type_str);

    ov::Shape shape;
    stream & shape;

    std::size_t byte_size = 0;
    stream & byte_size;

    if (allocator) {
        var = allocator(type, shape);
    } else {
        var = ov::Tensor(type, shape);
    }
    NPUW_ASSERT(var && "Tensor allocator returned an empty tensor");
    NPUW_ASSERT(var.get_element_type() == type && var.get_shape() == shape &&
                "Tensor allocator returned tensor with unexpected type or shape");
    NPUW_ASSERT(var.get_byte_size() == byte_size && "Tensor allocator returned tensor with unexpected byte size");

    stream.bytes(var.data(), byte_size);
}

void ov::npuw::s11n::serialize(Stream& stream, std::shared_ptr<ov::op::v0::Parameter>& var) {
    if (stream.input()) {
        std::string elem_type_str;
        std::string part_shape_str;
        std::unordered_set<std::string> names;
        stream & elem_type_str & part_shape_str & names;
        var = std::make_shared<op::v0::Parameter>(ov::element::Type(elem_type_str), ov::PartialShape(part_shape_str));
        if (!names.empty()) {
            var->set_friendly_name(*names.begin());
        }
        var->output(0).get_tensor().set_names(names);
    } else {
        OPENVINO_THROW("Parameter pointer is read-only in NPUW serialization");
    }
}

void ov::npuw::s11n::serialize(Stream& stream, std::shared_ptr<ov::Node>& var) {
    if (stream.input()) {
        std::string elem_type_str;
        std::string part_shape_str;
        std::unordered_set<std::string> names;
        stream & elem_type_str & part_shape_str & names;
        std::shared_ptr<ov::Node> res =
            std::make_shared<ov::op::v0::Constant>(ov::element::Type(elem_type_str), std::vector<size_t>{1});
        const std::shared_ptr<ov::descriptor::Tensor>& tensor_dummy =
            std::make_shared<ov::descriptor::Tensor>(ov::element::Type(elem_type_str),
                                                     ov::PartialShape(part_shape_str),
                                                     names);
        var = std::make_shared<ov::op::v0::Result>(res);
        var->output(0).set_tensor_ptr(tensor_dummy);
        if (!names.empty()) {
            var->set_friendly_name(*names.begin());
        }
    } else {
        OPENVINO_THROW("Node pointer is read-only in NPUW serialization");
    }
}

void ov::npuw::s11n::serialize(Stream& stream, ov::Any& var) {
    std::string str;
    if (stream.output()) {
        str = ov::npuw::s11n::anyToString(var);
    }
    stream & str;
    if (stream.input()) {
        var = ov::npuw::s11n::stringToAny(str);
    }
}

void ov::npuw::s11n::serialize(Stream& stream, ov::CacheMode& var) {
    stream.bytes(&var, sizeof var);
}

void ov::npuw::s11n::serialize(Stream& stream, ov::element::Type& var) {
    stream.bytes(&var, sizeof var);
}

void ov::npuw::s11n::serialize(Stream& stream, ov::hint::PerformanceMode& var) {
    stream.bytes(&var, sizeof var);
}

void ov::npuw::s11n::serialize(Stream& stream, ov::AnyMap& var) {
    std::string str;
    if (stream.output()) {
        str = ov::npuw::s11n::anyMapToString(var);
    }
    stream & str;
    if (stream.input()) {
        var = ov::npuw::s11n::stringToAnyMap(str);
    }
}

// Weightless
// FIXME: all serialization needs a good rewriting
void ov::npuw::s11n::serialize_weightless(Stream& stream,
                                          std::vector<ov::Tensor>& var,
                                          const ov::npuw::s11n::WeightsContext& ctx) {
    if (stream.output()) {
        auto size = var.size();
        serialize(stream, size);
        for (auto& t : var) {
            if (!t) {
                bool is_initialized = false;
                serialize(stream, is_initialized);
                continue;
            }
            bool is_initialized = true;
            serialize(stream, is_initialized);
            auto data = t.data();
            auto iter = ctx.const_to_offset.find(data);
            if (iter == ctx.const_to_offset.end()) {
                bool is_weightless = false;
                serialize(stream, is_weightless);
                auto tensor = t;
                serialize(stream, tensor);
            } else {
                bool is_weightless = true;
                serialize(stream, is_weightless);
                auto elem_type = t.get_element_type().to_string();
                serialize(stream, elem_type);
                auto shape = t.get_shape();
                serialize(stream, shape);
                auto byte_size = t.get_byte_size();
                serialize(stream, byte_size);
                auto offset = iter->second;
                serialize(stream, offset);
            }
        }
        return;
    }

    var.clear();
    std::size_t size;
    serialize(stream, size);
    for (std::size_t i = 0; i < size; ++i) {
        bool is_initialized = false;
        serialize(stream, is_initialized);
        if (!is_initialized) {
            var.push_back(ov::Tensor());
            continue;
        }
        bool is_weightless = false;
        serialize(stream, is_weightless);
        if (is_weightless) {
            std::string type_str;
            serialize(stream, type_str);
            ov::element::Type type(type_str);
            ov::Shape shape;
            serialize(stream, shape);
            std::size_t byte_size = 0;
            serialize(stream, byte_size);
            std::size_t offset = 0;
            serialize(stream, offset);
            ov::Tensor t(type, shape);

            if (ctx.weights) {
                if (ctx.bf16_consts.find({offset, byte_size}) != ctx.bf16_consts.end()) {
                    NPUW_ASSERT(type == ov::element::f16);
                    // Read original bf16 weight
                    auto bf16_tensor = ov::Tensor(ov::element::bf16, shape);
                    NPUW_ASSERT(bf16_tensor.get_byte_size() == byte_size);
                    std::memcpy(bf16_tensor.data(), ctx.weights->get_ptr(offset), byte_size);

                    NPUW_ASSERT(bf16_tensor.get_size() == t.get_size());

                    // Transform bf16 to f16 tensor
                    using dst_type = typename element_type_traits<ov::element::Type_t::f16>::value_type;
                    auto src_data = bf16_tensor.data<ov::bfloat16>();
                    auto dst_data = t.data<dst_type>();
                    ov::reference::convert_from_bf16_to_f16_with_clamp(src_data, dst_data, t.get_size());
                } else {
                    std::memcpy(t.data(), ctx.weights->get_ptr(offset), byte_size);
                }
            } else {
                auto it = ctx.consts_cache.find({offset, byte_size});
                NPUW_ASSERT(it != ctx.consts_cache.end() && "Couldn't find Constant in cache!");
                t = ov::npuw::util::copy_tensor_from_const(it->second);
                NPUW_ASSERT(t.get_byte_size() == byte_size && t.get_shape() == shape && t.get_element_type() == type);
            }

            var.push_back(t);
        } else {
            ov::Tensor t;
            serialize(stream, t);
            var.push_back(t);
        }
    }
}
