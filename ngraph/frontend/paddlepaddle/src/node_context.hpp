// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <ngraph/variant.hpp>
#include <paddlepaddle_frontend/exceptions.hpp>
#include <paddlepaddle_frontend/utility.hpp>

#define NGRAPH_VARIANT_DECLARATION(TYPE, info)                                                     \
    template <>                                                                                    \
    class VariantWrapper<TYPE> : public VariantImpl<TYPE>                                          \
    {                                                                                              \
    public:                                                                                        \
        static constexpr VariantTypeInfo type_info{info, 0};                                       \
        const VariantTypeInfo& get_type_info() const override { return type_info; }                \
        VariantWrapper<TYPE>(const value_type& value)                                              \
            : VariantImpl<value_type>(value)                                                       \
        {                                                                                          \
        }                                                                                          \
    }

namespace ngraph
{
    NGRAPH_VARIANT_DECLARATION(int32_t, "Variant::int32");
    NGRAPH_VARIANT_DECLARATION(std::vector<int32_t>, "Variant::int32_vector");
    NGRAPH_VARIANT_DECLARATION(float, "Variant::float");
    NGRAPH_VARIANT_DECLARATION(std::vector<float>, "Variant::float_vector");
    NGRAPH_VARIANT_DECLARATION(bool, "Variant::bool");
    NGRAPH_VARIANT_DECLARATION(ngraph::element::Type, "Variant::element_type");
    NGRAPH_VARIANT_DECLARATION(std::vector<int64_t>, "Variant::int64_vector");

    namespace frontend
    {
        namespace pdpd
        {
            using InPortName = std::string;
            using OutPortName = std::string;
            using TensorName = std::string;
            using NamedOutputs = std::map<OutPortName, OutputVector>;
            using NamedInputs = std::map<InPortName, OutputVector>;

            class DecoderBase
            {
            public:
                /// \brief Get attribute value by name and requested type
                ///
                /// \param name Attribute name
                /// \param type_info Attribute type information
                /// \return Shared pointer to appropriate value if it exists, 'nullptr' otherwise
                virtual std::shared_ptr<Variant>
                    get_attribute(const std::string& name,
                                  const VariantTypeInfo& type_info) const = 0;

                virtual std::vector<OutPortName> get_output_names() const = 0;

                /// \brief Get output port type
                ///
                /// Current API assumes that output port has only one output type.
                /// If decoder supports multiple types for specified port, it shall throw general
                /// exception
                ///
                /// \param port_name Port name for the node
                ///
                /// \return Type of specified output port
                virtual ngraph::element::Type
                    get_out_port_type(const std::string& port_name) const = 0;

                virtual std::string get_op_type() const = 0;
            };

            /// Keep necessary data for a single node in the original FW graph to facilitate
            /// conversion process in the rules code.
            class NodeContext
            {
                const DecoderBase& decoder;
                const NamedInputs& name_map;

            public:
                NodeContext(const DecoderBase& _decoder, const NamedInputs& _name_map)
                    : decoder(_decoder)
                    , name_map(_name_map)
                {
                }

                /// Returns node attribute by name. Returns 'def' value if attribute does not exist
                template <typename T>
                T get_attribute(const std::string& name, const T& def) const
                {
                    auto res = decoder.get_attribute(name, VariantWrapper<T>::type_info);
                    if (res)
                    {
                        auto ret = std::dynamic_pointer_cast<VariantWrapper<T>>(res);
                        FRONT_END_GENERAL_CHECK(
                            ret, "Attribute with name '", name, "' has invalid type");
                        return ret->get();
                    }
                    else
                    {
                        return def;
                    }
                }

                template <typename T>
                T get_attribute(const std::string& name) const
                {
                    auto res = decoder.get_attribute(name, VariantWrapper<T>::type_info);
                    FRONT_END_GENERAL_CHECK(res, "Attribute with name '", name, "' does not exist");
                    auto ret = std::dynamic_pointer_cast<VariantWrapper<T>>(res);
                    FRONT_END_GENERAL_CHECK(
                        ret, "Attribute with name '", name, "' has invalid type");
                    return ret->get();
                }

                template <typename T>
                bool has_attribute(const std::string& name) const
                {
                    return decoder.get_attribute(name, VariantWrapper<T>::type_info) != nullptr;
                }

                /// Detects if there is at least one input attached with a given name
                bool has_ng_input(const std::string& name) const
                {
                    auto found = name_map.find(name);
                    if (found != name_map.end())
                        return !found->second.empty();
                    return false;
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

                std::vector<OutPortName> get_output_names() const
                {
                    return decoder.get_output_names();
                }

                ngraph::element::Type get_out_port_type(const std::string& port_name) const
                {
                    return decoder.get_out_port_type(port_name);
                }

                std::string get_op_type() const { return decoder.get_op_type(); }

                NamedOutputs default_single_output_mapping(
                    const std::shared_ptr<Node>& ngraph_node,
                    const std::vector<OutPortName>& required_pdpd_out_names) const;
            };

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

        } // namespace pdpd
    }     // namespace frontend
} // namespace ngraph
