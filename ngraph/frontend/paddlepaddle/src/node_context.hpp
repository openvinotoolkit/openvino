// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <paddlepaddle_frontend/exceptions.hpp>
#include "decoder.hpp"

namespace ngraph
{
    namespace frontend
    {
        namespace pdpd
        {
            using InPortName = std::string;
            using OutPortName = std::string;
            using TensorName = std::string;
            using NamedOutputs = std::map<OutPortName, OutputVector>;
            using NamedInputs = std::map<InPortName, OutputVector>;

            /// Keep necessary data for a single node in the original FW graph to facilitate
            /// conversion process in the rules code.
            class NodeContext
            {
                const DecoderPDPDProto& node;
                const NamedInputs& name_map;

            public:
                NodeContext(const DecoderPDPDProto& _node, NamedInputs& _name_map)
                    : node(_node)
                    , name_map(_name_map)
                {
                }

                /// Detects if there is at least one input attached with a given name
                bool has_ng_input(const std::string& name) const
                {
                    auto found = name_map.find(name);
                    if (found != name_map.end())
                        return !found->second.empty();
                    return false;
                }

                size_t get_ng_input_size(const std::string& name) const
                {
                    return name_map.at(name).size();
                }

                /// Returns exactly one input with a given name; throws if there is no inputs or
                /// there are more than one input
                Output<Node> get_ng_input(const std::string& name) const
                {
                    FRONT_END_GENERAL_CHECK(name_map.at(name).size() == 1);
                    return name_map.at(name).at(0);
                }

                /// Returns all inputs with a given name
                OutputVector get_ng_inputs(const std::string& name) const
                {
                    return name_map.at(name);
                }

                template <typename T>
                T get_attribute(const std::string& name, const T& def = T()) const;

                template <typename T>
                bool has_attribute(const std::string& name) const
                {
                    // TODO: Rework this hack
                    try
                    {
                        get_attribute<T>(name);
                        return true;
                    }
                    catch (const GeneralFailure&)
                    {
                        return false;
                    }
                }

                const std::string& op_type() const { return node.get_op_type(); }
                std::vector<OutPortName> get_output_names() const
                {
                    return node.get_output_names();
                }
                std::vector<ngraph::element::Type>
                    get_out_port_types(const std::string& port_name) const
                {
                    return node.get_out_port_types(port_name);
                }
                ngraph::element::Type get_out_port_type(const std::string& port_name) const;
                NamedOutputs default_single_output_mapping(
                    const std::shared_ptr<Node>& ngraph_node,
                    const std::vector<OutPortName>& required_pdpd_out_names) const;
            };

            template <>
            inline int32_t NodeContext::get_attribute(const std::string& name,
                                                      const int32_t& def) const
            {
                return node.get_int(name, def);
            }

            template <>
            inline float NodeContext::get_attribute(const std::string& name, const float& def) const
            {
                return node.get_float(name, def);
            }

            template <>
            inline std::string NodeContext::get_attribute(const std::string& name,
                                                          const std::string& def) const
            {
                return node.get_str(name, def);
            }

            template <>
            inline std::vector<int32_t>
                NodeContext::get_attribute(const std::string& name,
                                           const std::vector<int32_t>& def) const
            {
                return node.get_ints(name, def);
            }

            template <>
            inline std::vector<float>
                NodeContext::get_attribute(const std::string& name,
                                           const std::vector<float>& def) const
            {
                return node.get_floats(name, def);
            }

            template <>
            inline bool NodeContext::get_attribute(const std::string& name, const bool& def) const
            {
                return node.get_bool(name, def);
            }

            template <>
            inline ngraph::element::Type
                NodeContext::get_attribute(const std::string& name,
                                           const ngraph::element::Type& def) const
            {
                return node.get_dtype(name, def);
            }

            inline ngraph::element::Type
                NodeContext::get_out_port_type(const std::string& port_name) const
            {
                auto types = get_out_port_types(port_name);
                FRONT_END_GENERAL_CHECK(types.size() > 0, "Port has no tensors connected.");
                FRONT_END_GENERAL_CHECK(std::equal(types.begin() + 1, types.end(), types.begin()),
                                        "Port has tensors with different types connected.");
                return types[0];
            }

            inline NamedOutputs NodeContext::default_single_output_mapping(
                const std::shared_ptr<Node>& ngraph_node,
                const std::vector<OutPortName>& required_pdpd_out_names) const
            {
                NamedOutputs named_outputs;
                const auto& ngraph_outputs = ngraph_node->outputs();
                const auto& pdpd_op_output_names = this->get_output_names();
                FRONT_END_GENERAL_CHECK(ngraph_outputs.size() == 1,
                                        "nGraph node must have exactly one output");
                for (const auto& pdpd_name : pdpd_op_output_names)
                {
                    if (std::find(required_pdpd_out_names.begin(),
                                  required_pdpd_out_names.end(),
                                  pdpd_name) != required_pdpd_out_names.end())
                        named_outputs[pdpd_name] = {ngraph_outputs[0]};
                }
                return named_outputs;
            }
            template <>
            inline std::vector<int64_t>
                NodeContext::get_attribute(const std::string& name,
                                           const std::vector<int64_t>& def) const
            {
                return node.get_longs(name, def);
            }

            template <>
            inline int64_t NodeContext::get_attribute(const std::string& name,
                                                      const int64_t& def) const
            {
                return node.get_long(name, def);
            }

        } // namespace pdpd
    }     // namespace frontend
} // namespace ngraph
