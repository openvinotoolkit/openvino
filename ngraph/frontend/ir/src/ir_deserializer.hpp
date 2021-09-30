// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <xml_parse_utils.h>

#include <ie_ngraph_utils.hpp>
#include <ir_frontend/model.hpp>
#include <istream>
#include <memory>
#include <ngraph/ngraph.hpp>
#include <pugixml.hpp>
#include <utils.hpp>

namespace ov {
struct GenericLayerParams {
    struct LayerPortData {
        size_t portId;
        std::vector<ngraph::Dimension> dims;
        ngraph::element::Type_t precision;
        std::unordered_set<std::string> names;
    };
    size_t layerId;
    std::string version;
    std::string name;
    std::string type;
    std::vector<LayerPortData> inputPorts;
    std::vector<LayerPortData> outputPorts;

    size_t getRealInputPortId(size_t id) const {
        size_t real_id = 0;
        for (auto& it : inputPorts) {
            if (it.portId == id) {
                return real_id;
            }
            ++real_id;
        }
        IE_THROW() << "Can not find input port with id " << id << " in layer " << name;
    }

    size_t getRealOutputPortId(size_t id) const {
        size_t real_id = 0;
        for (auto& it : outputPorts) {
            if (it.portId == id) {
                return real_id;
            }
            ++real_id;
        }
        IE_THROW() << "Can not find output port with id " << id << " in layer " << name;
    }
};

class XmlDeserializer : public ngraph::AttributeVisitor {
public:
    explicit XmlDeserializer(const pugi::xml_node& node,
                             const ov::Weights& weights,
                             const std::unordered_map<std::string, ngraph::OpSet>& opsets,
                             std::unordered_map<std::string, std::shared_ptr<ngraph::Variable>>& variables)
        : m_node(node),
          m_weights(weights),
          m_opsets(opsets),
          m_variables(variables) {}

    void on_adapter(const std::string& name, ngraph::ValueAccessor<std::string>& value) override {
        std::string val;
        if (!getStrAttribute(m_node.child("data"), name, val))
            return;
        value.set(val);
    }
    void on_adapter(const std::string& name, ngraph::ValueAccessor<bool>& value) override {
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
    void on_adapter(const std::string& name, ngraph::ValueAccessor<void>& adapter) override;

    void on_adapter(const std::string& name, ngraph::ValueAccessor<double>& adapter) override {
        std::string val;
        if (!getStrAttribute(m_node.child("data"), name, val))
            return;
        adapter.set(stringToType<double>(val));
    }
    void on_adapter(const std::string& name, ngraph::ValueAccessor<int64_t>& adapter) override {
        std::string val;
        if (!getStrAttribute(m_node.child("data"), name, val))
            return;
        adapter.set(stringToType<int64_t>(val));
    }

    void on_adapter(const std::string& name,
                    ngraph::ValueAccessor<std::shared_ptr<ngraph::Function>>& adapter) override;

    void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<int32_t>>& adapter) override {
        std::vector<int32_t> value;
        if (!getParameters<int32_t>(m_node.child("data"), name, value))
            return;
        adapter.set(value);
    }

    void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<int64_t>>& adapter) override {
        std::vector<int64_t> value;
        if (!getParameters<int64_t>(m_node.child("data"), name, value))
            return;
        adapter.set(value);
    }

    void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<float>>& adapter) override {
        std::vector<float> value;
        if (!getParameters<float>(m_node.child("data"), name, value))
            return;
        adapter.set(value);
    }

    void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<std::string>>& adapter) override {
        std::vector<std::string> value;
        if (!getParameters<std::string>(m_node.child("data"), name, value))
            return;
        adapter.set(value);
    }

    void use_framework_node(bool flag) {
        m_use_framework_node = flag;
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
    std::vector<std::shared_ptr<ngraph::op::util::SubGraphOp::InputDescription>> parseInputDescription(
        const pugi::xml_node& node);
    /// \brief Traverses port_map in order to create vector of OutputDescription shared_ptrs.
    /// Shall be used only for ops which have port_map attribute.
    /// \param node xml op representation
    std::vector<std::shared_ptr<ngraph::op::util::SubGraphOp::OutputDescription>> parseOutputDescription(
        const pugi::xml_node& node);

    // TODO consider to call only once per layer/TI-Loop node
    IoMap updated_io_map(const pugi::xml_node& node);

    /// \brief Traverses xml node representation in order to create nGraph function for it.
    /// \param node xml node representation
    /// \param weights weights attached to current node
    /// \return shared pointer to function representing input node
    std::shared_ptr<ngraph::Function> parse_function(const pugi::xml_node& root, const ov::Weights& weights);
    /// \brief Traverses xml node representation in order to get the purpose attribute of
    /// inputs/outputs in the body of Loop op. \param node xml node representation \return struct
    /// with value of purpuse attribute
    ngraph::op::v5::Loop::SpecialBodyPorts parsePurposeAttribute(const pugi::xml_node& node);

    GenericLayerParams parseGenericParams(const pugi::xml_node& node);

    std::shared_ptr<ngraph::Node> createNode(const ngraph::OutputVector& inputs,
                                             const pugi::xml_node& node,
                                             const ov::Weights& weights,
                                             const GenericLayerParams& params);

    // -- DATA --
    const pugi::xml_node m_node;
    const ov::Weights& m_weights;
    const std::unordered_map<std::string, ngraph::OpSet>& m_opsets;
    std::unordered_map<std::string, std::shared_ptr<ngraph::Variable>>& m_variables;

    ///
    /// store information about parameters/results order during function creation
    /// it will be used during Inputs/Outputs Description creation in SubGraph processing
    ///
    IoMap io_map;

    bool m_use_framework_node{false};
};
}  // namespace ov