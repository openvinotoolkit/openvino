// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cctype>
#include <istream>
#include <memory>
#include <pugixml.hpp>

#include "input_model.hpp"
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/op_extension.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/util/sub_graph_base.hpp"
#include "openvino/runtime/aligned_buffer.hpp"
#include "utils.hpp"

namespace ov {

struct GenericLayerParams {
    struct LayerPortData {
        size_t portId;
        std::vector<ov::Dimension> dims;
        ov::element::Type_t precision;
        std::unordered_set<std::string> names;
    };
    size_t layerId;
    std::string version;
    std::string name;
    std::string type;
    std::vector<LayerPortData> inputPorts;
    std::vector<LayerPortData> outputPorts;

    size_t get_real_input_port_id(size_t id) const {
        size_t real_id = 0;
        for (auto& it : inputPorts) {
            if (it.portId == id) {
                return real_id;
            }
            ++real_id;
        }
        OPENVINO_THROW("Can not find input port with id ", id, " in layer ", name);
    }

    size_t get_real_output_port_id(size_t id) const {
        size_t real_id = 0;
        for (auto& it : outputPorts) {
            if (it.portId == id) {
                return real_id;
            }
            ++real_id;
        }
        OPENVINO_THROW("Can not find output port with id ", id, " in layer ", name);
    }
};

class XmlDeserializer : public ov::AttributeVisitor {
public:
    explicit XmlDeserializer(const pugi::xml_node& node,
                             const std::shared_ptr<ov::AlignedBuffer>& weights,
                             const std::unordered_map<std::string, ov::OpSet>& opsets,
                             const std::unordered_map<ov::DiscreteTypeInfo, ov::BaseOpExtension::Ptr>& extensions,
                             std::unordered_map<std::string, std::shared_ptr<ov::op::util::Variable>>& variables,
                             size_t version)
        : m_node(node),
          m_weights(weights),
          m_opsets(opsets),
          m_extensions(extensions),
          m_variables(variables),
          m_version(version) {}

    void on_adapter(const std::string& name, ov::ValueAccessor<std::string>& value) override {
        std::string val;
        if (!getStrAttribute(m_node.child("data"), name, val))
            return;
        value.set(val);
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<bool>& value) override {
        std::string val;
        if (!getStrAttribute(m_node.child("data"), name, val))
            return;
        std::transform(val.begin(), val.end(), val.begin(), [](char ch) {
            return std::tolower(static_cast<unsigned char>(ch));
        });
        std::set<std::string> true_names{"true", "1"};
        std::set<std::string> false_names{"false", "0"};

        bool is_true = true_names.find(val) != true_names.end();
        bool is_false = false_names.find(val) != false_names.end();

        if (!is_true && !is_false)
            return;
        value.set(is_true);
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<void>& adapter) override;

    void on_adapter(const std::string& name, ov::ValueAccessor<double>& adapter) override {
        std::string val;
        if (!getStrAttribute(m_node.child("data"), name, val))
            return;
        adapter.set(stringToType<double>(val));
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<int64_t>& adapter) override {
        std::string val;
        if (!getStrAttribute(m_node.child("data"), name, val))
            return;
        adapter.set(stringToType<int64_t>(val));
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<std::shared_ptr<ov::Model>>& adapter) override;

    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<int32_t>>& adapter) override {
        std::vector<int32_t> value;
        if (!getParameters<int32_t>(m_node.child("data"), name, value))
            return;
        adapter.set(value);
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<int64_t>>& adapter) override {
        std::vector<int64_t> value;
        if (!getParameters<int64_t>(m_node.child("data"), name, value))
            return;
        adapter.set(value);
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<float>>& adapter) override {
        std::vector<float> value;
        if (!getParameters<float>(m_node.child("data"), name, value))
            return;
        adapter.set(value);
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<std::string>>& adapter) override {
        std::vector<std::string> value;
        if (!getParameters<std::string>(m_node.child("data"), name, value))
            return;
        adapter.set(value);
    }

private:
    struct IoMap {
        using NodeIdToIoIndex = std::unordered_map<size_t /*xml node id*/, uint64_t /*body io index*/>;
        NodeIdToIoIndex inputs;
        NodeIdToIoIndex outputs;
    };

    /// \brief Traverses port_map in order to create vector of InputDescription shared_ptrs.
    /// Shall be used only for ops which have port_map attribute.
    /// \param node xml op representation
    std::vector<std::shared_ptr<ov::op::util::SubGraphOp::InputDescription>>
    parse_input_description(const pugi::xml_node& node, const std::string& body_name, const std::string& port_map_name);
    /// \brief Traverses port_map in order to create vector of OutputDescription shared_ptrs.
    /// Shall be used only for ops which have port_map attribute.
    /// \param node xml op representation
    std::vector<std::shared_ptr<ov::op::util::SubGraphOp::OutputDescription>> parse_output_description(
        const pugi::xml_node& node,
        const std::string& body_name,
        const std::string& port_map_name);

    // TODO consider to call only once per layer/TI-Loop node
    IoMap updated_io_map(const pugi::xml_node& node, const pugi::xml_node& body_node);

    /// \brief Traverses xml node representation in order to create ov function for it.
    /// \param node xml node representation
    /// \param weights weights attached to current node
    /// \return shared pointer to function representing input node
    std::shared_ptr<ov::Model> parse_function(const pugi::xml_node& root,
                                              const std::shared_ptr<ov::AlignedBuffer>& weights);
    /// \brief Traverses xml node representation in order to get the purpose attribute of
    /// inputs/outputs in the body of Loop op. \param node xml node representation \return struct
    /// with value of purpuse attribute
    ov::op::v5::Loop::SpecialBodyPorts parse_purpose_attribute(const pugi::xml_node& node);

    GenericLayerParams parse_generic_params(const pugi::xml_node& node);

    std::shared_ptr<ov::Node> create_node(const ov::OutputVector& inputs,
                                          const pugi::xml_node& node,
                                          const std::shared_ptr<ov::AlignedBuffer>& weights,
                                          const GenericLayerParams& params);

    void read_meta_data(const std::shared_ptr<ov::Model>& model, const pugi::xml_node& meta_section);

    void read_legacy_meta_data(const std::shared_ptr<ov::Model>& model,
                               const std::unordered_set<std::string>& names,
                               const pugi::xml_node& root_section);

    // -- DATA --
    const pugi::xml_node m_node;
    const std::shared_ptr<ov::AlignedBuffer>& m_weights;
    const std::unordered_map<std::string, ov::OpSet>& m_opsets;
    const std::unordered_map<ov::DiscreteTypeInfo, ov::BaseOpExtension::Ptr>& m_extensions;
    std::unordered_map<std::string, std::shared_ptr<ov::op::util::Variable>>& m_variables;

    ///
    /// store information about parameters/results order during a model creation
    /// it will be used during Inputs/Outputs Description creation in SubGraph processing
    ///
    IoMap io_map;

    int64_t m_version;
};
}  // namespace ov
