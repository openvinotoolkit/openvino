// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <map>
#include <pugixml.hpp>
#include <string>
#include <vector>

#include "openvino/core/op_extension.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/visibility.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/util/multi_subgraph_base.hpp"
#include "openvino/opsets/opset.hpp"
#include "openvino/runtime/aligned_buffer.hpp"

namespace ov::util {
struct GenericLayerParams;

class XmlDeserializer : public ov::AttributeVisitor {
public:
    explicit XmlDeserializer(const pugi::xml_node& node,
                             const std::shared_ptr<ov::AlignedBuffer>& weights,
                             const std::unordered_map<std::string, ov::OpSet>& opsets,
                             const std::unordered_map<ov::DiscreteTypeInfo, ov::BaseOpExtension::Ptr>& extensions,
                             std::unordered_map<std::string, std::shared_ptr<ov::op::util::Variable>>& variables,
                             size_t version);

    void on_adapter(const std::string& name, ov::ValueAccessor<std::string>& value) override;

    void on_adapter(const std::string& name, ov::ValueAccessor<bool>& value) override;
    void on_adapter(const std::string& name, ov::ValueAccessor<void>& adapter) override;

    void on_adapter(const std::string& name, ov::ValueAccessor<double>& adapter) override;
    void on_adapter(const std::string& name, ov::ValueAccessor<int64_t>& adapter) override;

    void on_adapter(const std::string& name, ov::ValueAccessor<std::shared_ptr<ov::Model>>& adapter) override;

    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<int32_t>>& adapter) override;

    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<int64_t>>& adapter) override;

    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<float>>& adapter) override;

    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<std::string>>& adapter) override;

protected:
    virtual ov::Any parse_weightless_cache_attribute(const pugi::xml_node& node) const;
    virtual void set_constant_num_buffer(ov::AttributeAdapter<std::shared_ptr<ov::AlignedBuffer>>& adapter);

    const pugi::xml_node& get_node() const;

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

    virtual std::unique_ptr<XmlDeserializer> make_visitor(
        const pugi::xml_node& node,
        const std::shared_ptr<ov::AlignedBuffer>& weights,
        const std::unordered_map<std::string, ov::OpSet>& opsets,
        const std::unordered_map<ov::DiscreteTypeInfo, ov::BaseOpExtension::Ptr>& extensions,
        std::unordered_map<std::string, std::shared_ptr<ov::op::util::Variable>>& variables,
        size_t version) const {
        return std::make_unique<XmlDeserializer>(node, weights, opsets, extensions, variables, version);
    }

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

}  // namespace ov::util
