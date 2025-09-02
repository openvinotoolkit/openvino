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
                                std::unordered_map<std::string, std::shared_ptr<ov::op::util::Variable>>& variables,
                                size_t version);

    void on_adapter(const std::string& name, ov::ValueAccessor<void>& adapter) override;

private:
    struct IoMap {
        using NodeIdToIoIndex = std::unordered_map<size_t /*xml node id*/, uint64_t /*body io index*/>;
        NodeIdToIoIndex inputs;
        NodeIdToIoIndex outputs;
    };

    std::vector<std::shared_ptr<ov::op::util::SubGraphOp::InputDescription>>
    parse_input_description(const pugi::xml_node& node, const std::string& body_name, const std::string& port_map_name);

    std::vector<std::shared_ptr<ov::op::util::MultiSubGraphOp::OutputDescription>> parse_output_description(
        const pugi::xml_node& node,
        const std::string& body_name,
        const std::string& port_map_name);

    ov::op::v5::Loop::SpecialBodyPorts parse_purpose_attribute(const pugi::xml_node& node);

    IoMap updated_io_map(const pugi::xml_node& node, const pugi::xml_node& body_node);
};

std::shared_ptr<ov::Model> deserialize_ir_model(std::string_view serialized_graph, const ov::Tensor& weights);

}  // namespace intel_npu
