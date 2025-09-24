// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/xml_util/xml_deserialize_util.hpp"

namespace intel_npu {

class NPUXmlDeserializer : public ov::util::XmlDeserializer {
public:
    explicit NPUXmlDeserializer(const pugi::xml_node& node,
                                const std::shared_ptr<ov::AlignedBuffer>& weights,
                                const std::unordered_map<std::string, ov::OpSet>& opsets,
                                const std::unordered_map<ov::DiscreteTypeInfo, ov::BaseOpExtension::Ptr>& extensions,
                                std::unordered_map<std::string, std::shared_ptr<ov::op::util::Variable>>& variables,
                                size_t version);
};

std::shared_ptr<ov::Model> deserialize_ir_model(uint8_t* serialized_model);

}  // namespace intel_npu
