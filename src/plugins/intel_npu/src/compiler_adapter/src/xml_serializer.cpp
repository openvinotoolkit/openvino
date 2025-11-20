// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "xml_serializer.hpp"

#include "intel_npu/weights_pointer_attribute.hpp"
#include "openvino/op/util/op_types.hpp"

namespace intel_npu {

ov::util::ConstantWriter& XmlSerializer::get_constant_write_handler() {
    return *m_weightless_constant_writer;
}

std::unique_ptr<ov::util::XmlSerializer> XmlSerializer::make_visitor(pugi::xml_node& data,
                                                                     const std::string& node_type_name,
                                                                     ov::util::ConstantWriter& constant_write_handler,
                                                                     int64_t version,
                                                                     bool,
                                                                     bool,
                                                                     ov::element::Type,
                                                                     bool) const {
    return std::make_unique<XmlSerializer>(data,
                                           node_type_name,
                                           constant_write_handler,
                                           version,
                                           m_weightless_constant_writer);
}

}  // namespace intel_npu
