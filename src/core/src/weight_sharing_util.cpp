// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/weight_sharing_util.hpp"

#include "openvino/core/model.hpp"
#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/runtime/aligned_buffer.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "openvino/util/mmap_object.hpp"
#include "openvino/util/variant_visitor.hpp"

namespace {

bool set_source_buffer(ov::wsh::WeightSourceRegistry& sources, const ov::wsh::WeightBuffer& buffer) {
    const auto& desc = buffer ? buffer->get_descriptor() : nullptr;
    const auto source_id = desc ? desc->get_id() : ov::wsh::invalid_source_id;
    if (source_id != ov::wsh::invalid_source_id) {
        sources[source_id] = buffer;
        return true;
    } else {
        return false;
    }
}

const ov::wsh::WeightMetaData* get_constant_meta(const ov::wsh::WeightRegistry& constants,
                                                 ov::wsh::DataID src_id,
                                                 ov::wsh::DataID id) {
    if (auto found_map = constants.find(src_id); found_map != constants.end()) {
        if (auto found_desc = found_map->second.find(id); found_desc != found_map->second.end()) {
            return &found_desc->second;
        }
    }
    return nullptr;
};
}  // namespace

namespace ov::weight_sharing {

bool Extension::set_constant_in_weight_registry(WeightRegistry& weight_registry, const ov::op::v0::Constant& constant) {
    if (const auto desc = constant.m_data->get_descriptor()) {
        const auto c_id = Extension::get_constant_id(constant);
        weight_registry[desc->get_id()][c_id] =
            WeightMetaData{desc->get_offset(), constant.get_byte_size(), constant.get_element_type()};
        return true;
    }
    return false;
}

WeightSourceRegistry Extension::get_weight_sources(const Model& model) {
    WeightSourceRegistry src_map;
    for (const auto& node : model.get_ops()) {
        if (const auto const_node = ov::as_type<ov::op::v0::Constant>(node.get())) {
            if (const auto desc = const_node->m_data->get_descriptor()) {
                if (auto src_buffer = desc->get_source_buffer()) {
                    src_map.emplace(desc->get_id(), std::move(src_buffer));
                }
            }
        }
    }
    return src_map;
}

WeightRegistry Extension::get_weight_registry(const Model& model) {
    WeightRegistry constant_map;
    for (const auto& node : model.get_ops()) {
        if (const auto const_node = ov::as_type<ov::op::v0::Constant>(node.get())) {
            set_constant_in_weight_registry(constant_map, *const_node);
        }
    }
    return constant_map;
}

DataID Extension::get_constant_source_id(const ov::op::v0::Constant& constant) {
    const auto desc = constant.m_data->get_descriptor();
    return desc ? desc->get_id() : invalid_source_id;
}

DataID Extension::get_constant_id(const ov::op::v0::Constant& constant) {
    const auto desc = constant.m_data->get_descriptor();
    return desc ? desc->get_offset() : invalid_constant_id;
}

std::optional<WeightOriginMetaData> Extension::get_constant_origin(const ov::op::v0::Constant& constant) {
    if (auto wl_attr = constant.get_rt_info().find(ov::WeightlessCacheAttribute::get_type_info_static());
        wl_attr != constant.get_rt_info().end()) {
        const auto& wl = wl_attr->second.as<ov::WeightlessCacheAttribute>();
        return std::make_optional(
            WeightOriginMetaData{get_constant_source_id(constant), wl.bin_offset, wl.original_size, wl.original_dtype});
    } else {
        return std::nullopt;
    }
}

std::shared_ptr<ov::AlignedBuffer> Extension::get_constant_source_buffer(const ov::op::v0::Constant& constant) {
    const auto desc = constant.m_data->get_descriptor();
    return desc ? desc->get_source_buffer() : nullptr;
}

std::shared_ptr<ov::AlignedBuffer> get_source_buffer(const Context& shared_context, const DataID source_id) {
    const auto& weights = shared_context.m_cache_sources;
    if (auto weight_it = weights.find(source_id); weight_it != weights.end()) {
        if (auto source_buffer = weight_it->second.lock()) {
            return std::make_shared<ov::SharedBuffer<WeightBuffer>>(source_buffer->get_ptr<char>(),
                                                                    source_buffer->size(),
                                                                    source_buffer);
        }
    }
    return nullptr;
}

std::shared_ptr<ov::AlignedBuffer> get_buffer(const Context& shared_context,
                                              const DataID source_id,
                                              const DataID constant_id) {
    if (const auto source_it = shared_context.m_cache_sources.find(source_id);
        source_it != shared_context.m_cache_sources.end()) {
        if (auto wt_buffer = source_it->second.lock()) {
            if (auto meta = get_constant_meta(shared_context.m_weight_registry, source_id, constant_id)) {
                return std::make_shared<ov::SharedBuffer<WeightBuffer>>(wt_buffer->get_ptr<char>() + meta->m_offset,
                                                                        meta->m_size,
                                                                        wt_buffer);
            }
        }
    }
    return nullptr;
}

std::shared_ptr<ov::AlignedBuffer> get_buffer(const Context& shared_context,
                                              const std::shared_ptr<ov::AlignedBuffer>& source_buffer,
                                              const DataID constant_id) {
    if (source_buffer) {
        const auto& desc = source_buffer->get_descriptor();
        const auto wt_id = desc ? desc->get_id() : ov::wsh::invalid_source_id;
        const auto& meta_data = get_constant_meta(shared_context.m_weight_registry, wt_id, constant_id);
        if (meta_data) {
            return std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::AlignedBuffer>>>(
                source_buffer->get_ptr<char>() + meta_data->m_offset,
                meta_data->m_size,
                source_buffer);
        }
    }
    return nullptr;
}

std::shared_ptr<ov::AlignedBuffer> get_buffer(const Context& shared_context,
                                              const std::shared_ptr<ov::MappedMemory>& source_buffer,
                                              const DataID constant_id) {
    if (source_buffer) {
        const auto wt_id = source_buffer ? source_buffer->get_id() : ov::wsh::invalid_source_id;
        const auto& meta_data = get_constant_meta(shared_context.m_weight_registry, wt_id, constant_id);
        if (meta_data) {
            return std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::MappedMemory>>>(
                source_buffer->data() + meta_data->m_offset,
                meta_data->m_size,
                source_buffer);
        }
    }
    return nullptr;
}

bool set_constant(Context& shared_context, const ov::op::v0::Constant& constant) {
    return Extension::set_constant_in_weight_registry(shared_context.m_weight_registry, constant);
}

bool set_weight_source(Context& shared_context, const WeightBuffer& source_buffer) {
    return set_source_buffer(shared_context.m_cache_sources, source_buffer);
}

bool set_weight_source(Context& shared_context, const ov::op::v0::Constant& constant) {
    const auto source_buffer = Extension::get_constant_source_buffer(constant);
    return set_source_buffer(shared_context.m_cache_sources, source_buffer);
}

bool set_runtime_weight_source(Context& shared_context, const WeightBuffer& source_buffer) {
    return set_source_buffer(shared_context.m_runtime_sources, source_buffer);
}
}  // namespace ov::weight_sharing
