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
class BaseMmapDescriptor final : public ov::IBufferDescriptor {
public:
    BaseMmapDescriptor(std::shared_ptr<ov::MappedMemory> mem)
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
                std::make_shared<BaseMmapDescriptor>(mem));
        } else {
            return nullptr;
        }
    }

private:
    uint64_t m_id;
    std::weak_ptr<ov::MappedMemory> m_mem;
};

class BasicDescriptor final : public ov::IBufferDescriptor {
public:
    BasicDescriptor(uint64_t offset, const std::shared_ptr<ov::AlignedBuffer>& buffer)
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
        if (auto buffer = m_buffer.lock()) {
            return buffer;
        } else {
            return nullptr;
        }
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
                                 return std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::MappedMemory>>>(
                                     buffer->data(),
                                     buffer->size(),
                                     buffer,
                                     std::make_shared<BaseMmapDescriptor>(buffer));
                             }};

std::shared_ptr<ov::AlignedBuffer> get_shared_buffer(const ov::weight_sharing::WeakWeightBuffer& weak_buffer) {
    ov::util::VariantVisitor get_shared_buffer{[](const auto& mem) -> std::shared_ptr<ov::AlignedBuffer> {
        if (auto locked_mem = mem.lock()) {
            return get_aligned_buffer(locked_mem);
        }
        return nullptr;
    }};
    return std::visit(get_shared_buffer, weak_buffer);
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
                // naive implementation buffer should provide tag

                src_map.emplace(std::make_pair(desc->get_id(), const_node->m_data));
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

uint64_t Extension::get_constant_source_id(const ov::op::v0::Constant& constant) {
    const auto desc = constant.m_data->get_descriptor();
    return desc ? desc->get_id() : INVALID_SOURCE_ID;
}

uint64_t Extension::get_constant_id(const ov::op::v0::Constant& constant) {
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
    if (auto it = shared_context.m_weight_sources.find(source_id); it != shared_context.m_weight_sources.end()) {
        return get_shared_buffer(it->second);
    } else {
        return nullptr;
    }
}

std::shared_ptr<ov::AlignedBuffer> get_constant_buffer(const Context& shared_context,
                                                       const DataID source_id,
                                                       const DataID constant_id) {
    std::shared_ptr<ov::AlignedBuffer> constant_buffer;

    if (const auto source_it = shared_context.m_weight_sources.find(source_id);
        source_it != shared_context.m_weight_sources.end()) {
        if (auto buffer = get_shared_buffer(source_it->second)) {
            constant_buffer = get_constant_buffer(shared_context, buffer, constant_id);
        }
    }

    return constant_buffer;
}

std::shared_ptr<ov::AlignedBuffer> get_constant_buffer(const Context& shared_context,
                                                       const WeightBuffer& weight_buffer,
                                                       const DataID constant_id) {
    std::shared_ptr<ov::AlignedBuffer> constant_buffer;
    auto buff = std::visit(get_aligned_buffer, weight_buffer);

    if (auto desc = buff->get_descriptor()) {
        if (auto const_map_it = shared_context.m_constants_meta_data.find(desc->get_id());
            const_map_it != shared_context.m_constants_meta_data.end()) {
            if (auto const_desc = const_map_it->second.find(constant_id); const_desc != const_map_it->second.end()) {
                const auto& [c_offset, c_size, _] = const_desc->second;
                constant_buffer = std::make_shared<SharedBuffer<std::shared_ptr<AlignedBuffer>>>(
                    buff->get_ptr<char>() + c_offset,
                    c_size,
                    buff,
                    std::make_shared<BasicDescriptor>(c_offset, buff));
            }
        }
    }
    return constant_buffer;
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
