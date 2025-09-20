// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/xml_deserializer.hpp"

#include "openvino/runtime/shared_buffer.hpp"
#include "openvino/util/common_util.hpp"
#include "openvino/util/xml_parse_utils.hpp"

namespace intel_npu {

NPUXmlDeserializer::NPUXmlDeserializer(
    const pugi::xml_node& node,
    const std::shared_ptr<ov::AlignedBuffer>& weights,
    const std::unordered_map<std::string, ov::OpSet>& opsets,
    std::unordered_map<std::string, std::shared_ptr<ov::op::util::Variable>>& variables,
    size_t version)
    : ov::util::XmlDeserializer(node,
                                weights,
                                opsets,
                                std::unordered_map<ov::DiscreteTypeInfo, ov::BaseOpExtension::Ptr>(),
                                variables,
                                version) {}

std::shared_ptr<ov::Model> deserialize_ir_model(std::string_view serialized_graph, const ov::Tensor& weights) {
    ov::util::StringViewStreamBuf mb(serialized_graph);
    std::istream modelStream(&mb);
    pugi::xml_document m_xml_doc;
    pugi::xml_parse_result res = m_xml_doc.load(modelStream);
    OPENVINO_ASSERT(res.status == pugi::status_ok, res.description(), " at offset ", res.offset);
    pugi::xml_node root = m_xml_doc.document_element();

    std::shared_ptr<ov::AlignedBuffer> weights_buffer =
        std::make_shared<ov::SharedBuffer<ov::Tensor>>(reinterpret_cast<char*>(const_cast<void*>(weights.data())),
                                                       weights.get_byte_size(),
                                                       weights);

    std::unordered_map<std::string, ov::OpSet> opsets;
    for (const auto& it : ov::get_available_opsets()) {
        opsets[it.first] = it.second();
    }
    std::unordered_map<std::string, std::shared_ptr<ov::op::util::Variable>> variables;
    size_t version = static_cast<size_t>(ov::util::pugixml::get_uint64_attr(root, "version", 0));

    NPUXmlDeserializer visitor(root, weights_buffer, opsets, variables, version);
    std::shared_ptr<ov::Model> model;
    visitor.on_attribute("net", model);
    model->get_rt_info()["version"] = int64_t(version);

    return model;
}

}  // namespace intel_npu
