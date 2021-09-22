// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <ngraph/variant.hpp>
#include <tensorflow_frontend/exceptions.hpp>
#include <tensorflow_frontend/utility.hpp>
#include <tensorflow_frontend/place.hpp>

#include "types.pb.h"
#include "tensor.pb.h"

#define NGRAPH_VARIANT_DECLARATION(TYPE, info)                                            \
    template <>                                                                           \
    class VariantWrapper<TYPE> : public VariantImpl<TYPE> {                               \
    public:                                                                               \
        static constexpr VariantTypeInfo type_info{info, 0};                              \
        const VariantTypeInfo& get_type_info() const override {                           \
            return type_info;                                                             \
        }                                                                                 \
        VariantWrapper<TYPE>(const value_type& value) : VariantImpl<value_type>(value) {} \
    }

namespace ov {
NGRAPH_VARIANT_DECLARATION(int32_t, "Variant::int32");
NGRAPH_VARIANT_DECLARATION(std::vector<int32_t>, "Variant::int32_vector");
NGRAPH_VARIANT_DECLARATION(float, "Variant::float");
NGRAPH_VARIANT_DECLARATION(std::vector<float>, "Variant::float_vector");
NGRAPH_VARIANT_DECLARATION(bool, "Variant::bool");
NGRAPH_VARIANT_DECLARATION(ov::element::Type, "Variant::ov_element_type");
NGRAPH_VARIANT_DECLARATION(std::vector<int64_t>, "Variant::int64_vector");
NGRAPH_VARIANT_DECLARATION(ngraph::PartialShape, "Variant::ngraph_PartialShape");
NGRAPH_VARIANT_DECLARATION(std::vector<std::string>, "Variant::string_vector");
NGRAPH_VARIANT_DECLARATION(::tensorflow::DataType, "Variant::DataType");
NGRAPH_VARIANT_DECLARATION(::tensorflow::TensorProto, "Variant::TensorProto");
}  // namespace ov


namespace ngraph {
namespace frontend {
namespace tf {
using InPortName = size_t;
using OutPortName = size_t;
using NamedOutputs = std::map<OutPortName, OutputVector>;
using NamedInputs = std::map<InPortName, OutputVector>;

/// Keep necessary data for a single node in the original FW graph to facilitate
/// conversion process in the rules code.
class NodeContext {
    const ::ngraph::frontend::DecoderBase& decoder;
    const NamedInputs& name_map;
    std::map<std::string, ngraph::PartialShape> m_overridden_shapes;

public:
    NodeContext(const ::ngraph::frontend::DecoderBase& _decoder,
                const NamedInputs& _name_map)
        : decoder(_decoder),
          name_map(_name_map){}

    /// Returns node attribute by name. Returns 'def' value if attribute does not exist
    template <typename T>
    T get_attribute(const std::string& name, const T& def) const {
        auto res = decoder.get_attribute(name, VariantWrapper<T>::type_info);
        if (res) {
            auto ret = std::dynamic_pointer_cast<VariantWrapper<T>>(res);
            FRONT_END_GENERAL_CHECK(ret, "Attribute with name '", name, "' has invalid type");
            return ret->get();
        } else {
            return def;
        }
        return def;
    }

    template <typename T>
    T get_attribute(const std::string& name) const {
        auto res = decoder.get_attribute(name, VariantWrapper<T>::type_info);
        FRONT_END_GENERAL_CHECK(res, "Attribute with name '", name, "' does not exist");
        auto ret = std::dynamic_pointer_cast<VariantWrapper<T>>(res);
        FRONT_END_GENERAL_CHECK(ret, "Attribute with name '", name, "' has invalid type");
        return ret->get();
    }

    template <typename T>
    bool has_attribute(const std::string& name) const {
        return decoder.get_attribute(name, VariantWrapper<T>::type_info) != nullptr;
    }

    /// Detects if there is at least one input attached with a given name
    bool has_ng_input(const size_t& port_index) const {
        auto found = name_map.find(port_index);
        if (found != name_map.end())
            return !found->second.empty();
        return false;
    }

    /// Returns exactly one input with a given name; throws if there is no inputs or
    /// there are more than one input
    Output<Node> get_ng_input(const size_t& port_index) const {
        FRONT_END_GENERAL_CHECK(name_map.at(port_index).size() == 1);
        return name_map.at(port_index).at(0);
    }

    /// Returns all inputs with a given name
    OutputVector get_ng_inputs(const size_t& port_index) const {
        return name_map.at(port_index);
    }

    /// Returns all inputs in order they appear in map. This is used for FrameworkNode
    /// creation
    OutputVector get_all_ng_inputs() const {
        OutputVector res;
        for (const auto& entry : name_map) {
            res.insert(res.end(), entry.second.begin(), entry.second.end());
        }
        return res;
    }

    std::vector<OutPortName> get_output_names() const {
        return decoder.get_output_names();
    }
    
    ngraph::element::Type get_out_port_type(const size_t& port_index) const {
        return decoder.get_out_port_type(port_index);
    }

    std::string get_op_type() const {
        return decoder.get_op_type();
    }

    NamedOutputs default_single_output_mapping(const std::shared_ptr<Node>& ngraph_node,
                                               const std::vector<OutPortName>& required_tf_port_indices) const;

    // adds methods
    size_t get_ng_input_size() const {
        return name_map.size();
    }

    std::string get_name() const {
        return decoder.get_op_name();
    }
    
    std::vector<std::string> get_names() const {
        return {decoder.get_op_name()};
    }

    OutputVector get_ng_inputs() const {
        OutputVector res;
        for (const auto& entry : name_map) {
            res.insert(res.end(), entry.second.begin(), entry.second.end());
        }
        return res;
    }
    
    const std::map<std::string, ngraph::PartialShape>& get_overridden_shapes() const {
        return m_overridden_shapes;
    }
    
    const ::ngraph::frontend::DecoderBase* _get_decoder() const {
        return &decoder;
    }
};

inline NamedOutputs NodeContext::default_single_output_mapping(
    const std::shared_ptr<Node>& ngraph_node,
    const std::vector<OutPortName>& required_tf_port_indices) const {
    NamedOutputs named_outputs;
    const auto& ngraph_outputs = ngraph_node->outputs();
    const auto& tf_op_output_port_indices = this->get_output_names();
    FRONT_END_GENERAL_CHECK(ngraph_outputs.size() == 1, "nGraph node must have exactly one output");
    for (const auto& tf_port_index : tf_op_output_port_indices) {
        if (std::find(required_tf_port_indices.begin(), required_tf_port_indices.end(), tf_port_index) !=
            required_tf_port_indices.end())
            named_outputs[tf_port_index] = {ngraph_outputs[0]};
    }
    return named_outputs;
}

}  // namespace tf
}  // namespace frontend
}  // namespace ngraph
