// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "xml_serializer.hpp"

#include "intel_npu/weights_pointer_attribute.hpp"
#include "openvino/op/util/op_types.hpp"

namespace intel_npu {

ov::util::ConstantWriter& XmlSerializer::get_constant_write_handler() {
    if (m_use_weightless_writer) {
        return m_weightless_constant_writer;
    } else {
        return m_base_constant_writer;
    }
}

bool XmlSerializer::append_node_attributes(ov::Node& node) {
    // If the "WeightsPointerAttribute" is found, then we have the metadata required to avoid copying the weights
    // corresponding to this node.
    m_use_weightless_writer = node.get_rt_info().count(WeightsPointerAttribute::get_type_info_static()) != 0;
    auto result = ov::util::XmlSerializer::append_node_attributes(node);
    m_use_weightless_writer = false;
    return result;
}

std::unique_ptr<ov::util::XmlSerializer> XmlSerializer::make_visitor(pugi::xml_node& data,
                                                                     const std::string& node_type_name,
                                                                     ov::util::ConstantWriter& constant_write_handler,
                                                                     int64_t version,
                                                                     bool,
                                                                     bool,
                                                                     ov::element::Type,
                                                                     bool) const {
    return std::make_unique<XmlSerializer>(data, node_type_name, constant_write_handler, version);
}

}  // namespace intel_npu
