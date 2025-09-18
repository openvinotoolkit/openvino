// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/serialize.hpp"
#include "openvino/xml_util/xml_serialize_util.hpp"

namespace intel_npu {

class XmlSerializer : public ov::util::XmlSerializer {
public:
    XmlSerializer(pugi::xml_node& data,
                  const std::string& node_type_name,
                  ov::util::ConstantWriter& constant_write_handler,
                  int64_t version)
        : ov::util::XmlSerializer(data,
                                  node_type_name,
                                  constant_write_handler,
                                  version,
                                  false,
                                  false,
                                  ov::element::dynamic,
                                  false) {}
};

class StreamSerialize : public ov::pass::StreamSerialize {
public:
    StreamSerialize(std::ostream& stream,
                    ov::pass::Serialize::Version version = ov::pass::Serialize::Version::UNSPECIFIED)
        : ov::pass::StreamSerialize(stream, {}, {}, version) {}

private:
    std::unique_ptr<ov::util::XmlSerializer> make_serializer(pugi::xml_node& data,
                                                             const std::string& node_type_name,
                                                             ov::util::ConstantWriter& constant_write_handler,
                                                             int64_t version,
                                                             bool,
                                                             bool,
                                                             ov::element::Type,
                                                             bool) const override {
        return std::make_unique<XmlSerializer>(data, node_type_name, constant_write_handler, version);
    }
};

}  // namespace intel_npu
