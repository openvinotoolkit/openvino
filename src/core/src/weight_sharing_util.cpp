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
class MappedMemoryDescriptor final : public ov::IBufferDescriptor {
public:
    MappedMemoryDescriptor(std::shared_ptr<ov::MappedMemory> mem)
        : m_id(mem ? mem->get_id() : ov::weight_sharing::INVALID_SOURCE_ID),
          m_mem(mem) {}

    uint64_t get_id() const override {
        return m_id;
    }

    size_t get_offset() const override {
        return 0;
    }

    std::shared_ptr<ov::AlignedBuffer> get_source_buffer() const override {
        if (auto mem = m_mem.lock()) {
            return std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::MappedMemory>>>(
                mem->data(),
                mem->size(),
                mem,
                std::make_shared<MappedMemoryDescriptor>(mem));
        } else {
            return nullptr;
        }
    }

private:
    uint64_t m_id;
    std::weak_ptr<ov::MappedMemory> m_mem;
};

class SimpleDescriptor final : public ov::IBufferDescriptor {
public:
    SimpleDescriptor(uint64_t offset, const std::shared_ptr<ov::AlignedBuffer>& buffer)
        : m_id((buffer && buffer->get_descriptor()) ? buffer->get_descriptor()->get_id()
                                                    : ov::weight_sharing::INVALID_SOURCE_ID),
          m_offset(offset),
          m_buffer(buffer) {}

    uint64_t get_id() const override {
        return m_id;
    }

    size_t get_offset() const override {
        return m_offset;
    }

    std::shared_ptr<ov::AlignedBuffer> get_source_buffer() const override {
        return m_buffer.lock();
    }

private:
    uint64_t m_id;
    size_t m_offset;
    std::weak_ptr<ov::AlignedBuffer> m_buffer;
};

const auto get_aligned_buffer =
    ov::util::VariantVisitor{[](const auto& buffer) -> std::shared_ptr<ov::AlignedBuffer> {
                                 return buffer;
                             },
                             [](const std::shared_ptr<ov::MappedMemory>& buffer) -> std::shared_ptr<ov::AlignedBuffer> {
                                 if (buffer) {
                                     return std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::MappedMemory>>>(
                                         buffer->data(),
                                         buffer->size(),
                                         buffer,
                                         std::make_shared<MappedMemoryDescriptor>(buffer));
                                 } else {
                                     return nullptr;
                                 }
                             }};

std::shared_ptr<ov::AlignedBuffer> get_shared_buffer(const ov::weight_sharing::WeakWeightBuffer& weak_buffer) {
    return std::visit(ov::util::VariantVisitor{[](const auto& mem) -> std::shared_ptr<ov::AlignedBuffer> {
                          if (auto locked_mem = mem.lock()) {
                              return get_aligned_buffer(locked_mem);
                          }
                          return nullptr;
                      }},
                      weak_buffer);
}

const ov::wsh::ConstantMetaData* get_constant_meta(const ov::wsh::SourceConstantMap& constants,
                                                   ov::wsh::DataID src_id,
                                                   ov::wsh::DataID id) {
    if (auto found_map = constants.find(src_id); found_map != constants.end()) {
        if (auto found_desc = found_map->second.find(id); found_desc != found_map->second.end()) {
            return &found_desc->second;
        }
    }
    return nullptr;
};

const std::shared_ptr<ov::AlignedBuffer> make_const_from_context(const ov::wsh::SourceConstantMap& constants,
                                                                 const std::shared_ptr<ov::AlignedBuffer>& wt_buffer,
                                                                 ov::wsh::DataID id) {
    if (auto desc = wt_buffer->get_descriptor()) {
        if (const auto& meta = get_constant_meta(constants, desc->get_id(), id)) {
            return std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::AlignedBuffer>>>(
                wt_buffer->get_ptr<char>() + meta->m_offset,
                meta->m_size,
                wt_buffer,
                std::make_shared<SimpleDescriptor>(meta->m_offset, wt_buffer));
        }
    }
    return {};
}
}  // namespace

namespace ov::weight_sharing {

bool Extension::set_constant_in_constant_map(SourceConstantMap& constant_map, const ov::op::v0::Constant& constant) {
    if (const auto desc = constant.m_data->get_descriptor()) {
        constant_map[desc->get_id()][desc->get_offset()] =
            ConstantMetaData{desc->get_offset(), constant.get_byte_size(), constant.get_element_type()};
        return true;
    }
    return false;
}

WeightSourceMap Extension::make_weight_sources(const Model& model) {
    WeightSourceMap src_map;
    for (const auto& node : model.get_ops()) {
        if (const auto const_node = ov::as_type<ov::op::v0::Constant>(node.get())) {
            if (const auto desc = const_node->m_data->get_descriptor()) {
                src_map.emplace(desc->get_id(), desc->get_source_buffer());
            }
        }
    }
    return src_map;
}

SourceConstantMap Extension::make_constant_map(const Model& model) {
    SourceConstantMap constant_map;
    for (const auto& node : model.get_ops()) {
        if (const auto const_node = ov::as_type<ov::op::v0::Constant>(node.get())) {
            set_constant_in_constant_map(constant_map, *const_node);
        }
    }
    return constant_map;
}

DataID Extension::get_constant_source_id(const ov::op::v0::Constant& constant) {
    const auto desc = constant.m_data->get_descriptor();
    return desc ? desc->get_id() : INVALID_SOURCE_ID;
}

DataID Extension::get_constant_id(const ov::op::v0::Constant& constant) {
    const auto desc = constant.m_data->get_descriptor();
    return desc ? static_cast<int64_t>(desc->get_offset()) : INVALID_CONSTANT_ID;
}

std::optional<ConstantOriginMetaData> Extension::get_constant_origin(const ov::op::v0::Constant& constant) {
    if (auto wl_attr = constant.get_rt_info().find(ov::WeightlessCacheAttribute::get_type_info_static());
        wl_attr != constant.get_rt_info().end()) {
        const auto& wl = wl_attr->second.as<ov::WeightlessCacheAttribute>();
        return std::make_optional(ConstantOriginMetaData{get_constant_source_id(constant),
                                                         wl.bin_offset,
                                                         wl.original_size,
                                                         wl.original_dtype});
    } else {
        return std::nullopt;
    }
}

std::shared_ptr<ov::AlignedBuffer> get_source_buffer(const Context& shared_context, const DataID source_id) {
    const auto& weights = shared_context.m_weight_sources;
    if (auto weight_it = weights.find(source_id); weight_it != weights.end()) {
        return get_shared_buffer(weight_it->second);
    } else {
        return nullptr;
    }
}

std::shared_ptr<ov::AlignedBuffer> get_buffer(const Context& shared_context,
                                              const DataID source_id,
                                              const DataID constant_id) {
    if (const auto source_it = shared_context.m_weight_sources.find(source_id);
        source_it != shared_context.m_weight_sources.end()) {
        if (auto wt_buffer = get_shared_buffer(source_it->second)) {
            return make_const_from_context(shared_context.m_constants_meta_data, wt_buffer, constant_id);
        }
    }
    return nullptr;
}

std::shared_ptr<ov::AlignedBuffer> get_buffer(const Context& shared_context,
                                              const WeightBuffer& weight_buffer,
                                              const DataID constant_id) {
    const auto buffer = std::visit(get_aligned_buffer, weight_buffer);
    return buffer ? make_const_from_context(shared_context.m_constants_meta_data, buffer, constant_id) : nullptr;
}

bool set_constant(Context& shared_context, const ov::op::v0::Constant& constant) {
    return Extension::set_constant_in_constant_map(shared_context.m_constants_meta_data, constant);
}

bool set_weight_source(Context& shared_context, const WeightBuffer& constant) {
    ov::util::VariantVisitor set_weak_buffer{
        [&shared_context](const std::shared_ptr<ov::AlignedBuffer>& buffer) -> bool {
            if (buffer && buffer->get_descriptor() && buffer->get_descriptor()->get_id() != INVALID_SOURCE_ID) {
                shared_context.m_weight_sources[buffer->get_descriptor()->get_id()] = buffer;
                return true;
            }
            return false;
        },
        [&shared_context](const std::shared_ptr<ov::MappedMemory>& buffer) {
            if (buffer && buffer->get_id() != INVALID_SOURCE_ID) {
                shared_context.m_weight_sources[buffer->get_id()] = buffer;
                return true;
            } else {
                return false;
            }
        }};

    return std::visit(set_weak_buffer, constant);
}
}  // namespace ov::weight_sharing
