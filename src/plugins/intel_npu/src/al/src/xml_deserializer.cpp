// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/xml_deserializer.hpp"

#include "intel_npu/weights_pointer_attribute.hpp"
#include "openvino/op/group_query_attention.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "openvino/runtime/string_aligned_buffer.hpp"
#include "openvino/util/common_util.hpp"
#include "openvino/util/xml_parse_utils.hpp"
#include "ov_ops/rms.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"

namespace intel_npu {

NPUXmlDeserializer::NPUXmlDeserializer(
    const pugi::xml_node& node,
    const std::shared_ptr<ov::AlignedBuffer>& weights,
    const std::unordered_map<std::string, ov::OpSet>& opsets,
    const std::unordered_map<ov::DiscreteTypeInfo, ov::BaseOpExtension::Ptr>& extensions,
    std::unordered_map<std::string, std::shared_ptr<ov::op::util::Variable>>& variables,
    size_t version)
    : ov::util::XmlDeserializer(node, weights, opsets, extensions, variables, version) {}

ov::Any NPUXmlDeserializer::parse_weights_pointer_attribute(const pugi::xml_node& node) const {
    if (auto rt_info = node.child("rt_info")) {
        for (const auto& child : rt_info.children()) {
            for (const auto& attr : child.attributes()) {
                if (strcmp(attr.name(), "np") == 0) {
                    const auto ptr = reinterpret_cast<const void*>(ov::util::pugixml::get_uint64_attr(child, "np"));
                    const auto byte_size = ov::util::pugixml::get_uint64_attr(child, "ns");
                    return {WeightsPointerAttribute{ptr, byte_size}};
                }
            }
        }
    }
    return {};
}

void NPUXmlDeserializer::set_constant_num_buffer(ov::AttributeAdapter<std::shared_ptr<ov::AlignedBuffer>>& adapter) {
    const auto node = get_node();
    auto wl_attr = parse_weights_pointer_attribute(node);
    if (!wl_attr.is<WeightsPointerAttribute>()) {
        ov::util::XmlDeserializer::set_constant_num_buffer(adapter);
        return;
    }

    const auto& dn = node.child("data");
    const auto el_type = ov::element::Type(ov::util::pugixml::get_str_attr(dn, "element_type"));

    char* ptr = reinterpret_cast<char*>(wl_attr.as<WeightsPointerAttribute>().memory_pointer);
    size_t byte_size = wl_attr.as<WeightsPointerAttribute>().byte_size;

    std::shared_ptr<ov::AlignedBuffer> buffer;
    if (el_type != ov::element::string) {
        buffer = std::make_shared<ov::SharedBuffer<void*>>(ptr, byte_size, nullptr);
    } else {
        buffer = std::make_shared<ov::SharedStringAlignedBuffer>(ptr, byte_size);
    }
    adapter.set(buffer);
}

std::unique_ptr<ov::util::XmlDeserializer> NPUXmlDeserializer::make_visitor(
    const pugi::xml_node& node,
    const std::shared_ptr<ov::AlignedBuffer>& origin_weights,
    const std::unordered_map<std::string, ov::OpSet>& opsets,
    const std::unordered_map<ov::DiscreteTypeInfo, ov::BaseOpExtension::Ptr>& extensions,
    std::unordered_map<std::string, std::shared_ptr<ov::op::util::Variable>>& variables,
    size_t version) const {
    return std::make_unique<NPUXmlDeserializer>(node, origin_weights, opsets, extensions, variables, version);
}

std::shared_ptr<ov::Model> deserialize_ir_model(uint8_t* serialized_model) {
    ov::pass::StreamSerialize::DataHeader dataHeader;
    memcpy(&dataHeader, serialized_model, sizeof(dataHeader));

    pugi::xml_document xml_doc;
    pugi::xml_parse_result res = xml_doc.load_buffer(serialized_model + dataHeader.model_offset,
                                                     dataHeader.model_size,
                                                     pugi::parse_default,
                                                     pugi::encoding_utf8);
    ;
    OPENVINO_ASSERT(res.status == pugi::status_ok, res.description(), " at offset ", res.offset);
    pugi::xml_node root = xml_doc.document_element();

    std::shared_ptr<ov::AlignedBuffer> weights_buffer =
        std::make_shared<ov::SharedBuffer<void*>>(reinterpret_cast<char*>(serialized_model + dataHeader.consts_offset),
                                                  dataHeader.consts_size,
                                                  nullptr);

    std::unordered_map<std::string, ov::OpSet> opsets;
    for (const auto& it : ov::get_available_opsets()) {
        opsets[it.first] = it.second();
    }
    auto create_extensions_map = [&]() -> std::unordered_map<ov::DiscreteTypeInfo, ov::BaseOpExtension::Ptr> {
        std::unordered_map<ov::DiscreteTypeInfo, ov::BaseOpExtension::Ptr> extensions_map;
        const std::vector<ov::Extension::Ptr> extensions_vector{
            // TODO why these extensions?
            std::make_shared<ov::OpExtension<ov::op::internal::RMS>>(),
            std::make_shared<ov::OpExtension<ov::op::internal::RoPE>>(),
            std::make_shared<ov::OpExtension<ov::op::internal::GroupQueryAttention>>()};

        for (const auto& ext : extensions_vector) {
            if (auto base_ext = std::dynamic_pointer_cast<ov::BaseOpExtension>(ext))
                extensions_map.insert({base_ext->get_type_info(), base_ext});
        }
        return extensions_map;
    }();
    std::unordered_map<std::string, std::shared_ptr<ov::op::util::Variable>> variables;
    size_t version = static_cast<size_t>(ov::util::pugixml::get_uint64_attr(root, "version", 0));

    NPUXmlDeserializer visitor(root, weights_buffer, opsets, create_extensions_map, variables, version);
    std::shared_ptr<ov::Model> model;
    visitor.on_attribute("net", model);
    model->get_rt_info()["version"] = int64_t(version);

    return model;
}

}  // namespace intel_npu
