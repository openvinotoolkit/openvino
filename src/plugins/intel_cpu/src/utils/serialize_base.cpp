// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "serialize_base.hpp"

#include "openvino/core/descriptor_tensor.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/util/codec_xor.hpp"

namespace ov {
namespace intel_cpu {

ModelSerializer::ModelSerializer(std::ostream& ostream) : m_ostream(ostream) {}

void ModelSerializer::operator<<(const std::shared_ptr<ov::Model>& model) {
    auto serialize_info = [&](std::ostream& stream) {
        pugi::xml_document xml_doc;
        pugi::xml_node root = xml_doc.append_child("cnndata");
        pugi::xml_node outputs = root.append_child("outputs");
        for (const auto& out : model->get_results()) {
            auto out_node = outputs.append_child("out");
            auto name = ov::descriptor::get_ov_tensor_legacy_name(out->input_value(0).get_tensor());
            out_node.append_attribute("name").set_value(name.c_str());
        }
        xml_doc.save(stream);
    };

    ov::pass::StreamSerialize serializer(m_ostream, serialize_info, ov::util::codec_xor);
    serializer.run_on_model(std::const_pointer_cast<ov::Model>(model->clone()));
}

void ModelDeserializerBase::operator>>(std::shared_ptr<ov::Model>& model) {
    parse(model);
}

void ModelDeserializerBase::set_info(pugi::xml_node& root, std::shared_ptr<ov::Model>& model) {
    pugi::xml_node outputs = root.child("outputs");
    auto nodes_it = outputs.children("out").begin();
    size_t size = model->outputs().size();
    for (size_t i = 0lu; i < size; ++nodes_it, i++) {
        std::string name = nodes_it->attribute("name").value();
        if (name.empty())
            continue;
        auto result = model->output(i).get_node_shared_ptr();
        ov::descriptor::set_ov_tensor_legacy_name(result->input_value(0).get_tensor(), name);
    }
}

}   // namespace intel_cpu
}   // namespace ov
