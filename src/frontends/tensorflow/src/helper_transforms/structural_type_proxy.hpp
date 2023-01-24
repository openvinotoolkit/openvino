// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Purpose of this file is to define help proxies to represent structural data types that
// may be represented with multiple tensors in lowered-model. For example, tensor of strings
// can be represented as a single tensor if shape is scalar or as 3 tensors with it is not a scalar.
// Ragged tensors add two additional tensors to store indices for a ragged dimension. And so on --
// the exact representation depends on the type and will be defined explicitly.

// Helpers are defined as templated classes because they are used for description classes and for real values.

#pragma once

#include "openvino/core/type/non_tensor_type.hpp"
#include "openvino/frontend/tensorflow/visibility.hpp"


namespace ov {
namespace frontend {
namespace tensorflow {
namespace StructuralTypeProxy {


namespace StructuralType = element::StructuralType;


class Base {
public:

    Base (Any type) : m_type(type) {}

    virtual Any type () const {
        return m_type;
    }

    Any m_type;
};


template <typename PLowerTensorType>
class Tensor : public Base {
public:

    Tensor (const PLowerTensorType& tensor) : Base(StructuralType::Tensor(tensor->get_element_type())), m_tensor(tensor) {}
    PLowerTensorType m_tensor;
};


// Regular tensor with strings
template <typename PLowerTensorType>
class TensorStr : public Base {
public:

    TensorStr (const PLowerTensorType& begins, const PLowerTensorType& ends, const PLowerTensorType& values) :
        Base(StructuralType::Tensor(StructuralType::Str())), m_begins(begins), m_ends(ends), m_values(values)
    {}

    TensorStr (std::vector<PLowerTensorType>& inputs) :
        TensorStr(inputs[0], inputs[1], inputs[2])
    {}

    TensorStr (const PLowerTensorType& values) :
        Base(StructuralType::Tensor(StructuralType::Str())), m_values(values)
    {}

    PartialShape get_partial_shape () {
        return m_begins->get_partial_shape();
    }

    Shape get_shape () {
        return m_begins->get_shape();
    }

    std::string element_by_offset (size_t i) {
        const char* values = reinterpret_cast<char*>(m_values->template data<uint8_t>());
        if (m_begins) {
            auto begin = m_begins->template data<int32_t>()[i];
            auto end = m_ends->template data<int32_t>()[i];
            return std::string(values + begin, values + end);
        } else {
            return std::string(values, values + shape_size(m_values->get_shape()));
        }
    }

    PLowerTensorType m_begins = nullptr;
    PLowerTensorType m_ends = nullptr;
    PLowerTensorType m_values;
};


struct BindInput {
    BindInput (const std::vector<size_t>& _inputs, Any _structural_type) :
        inputs(_inputs), structural_type(_structural_type)
    {}

    BindInput (std::size_t start, std::size_t end, Any _structural_type) : structural_type(_structural_type) {
        for(size_t i = start; i < end; ++i)
            inputs.push_back(i);
    }

    std::vector<size_t> inputs;
    Any structural_type;
};


using BindInputs = std::vector<BindInput>;


inline std::vector<std::shared_ptr<Base>> structural_input_types (const std::vector<Input<Node>>& node_inputs, const BindInputs bindings) {
    std::vector<std::shared_ptr<Base>> result;
    // If there is structural types present in input tensors, they sould be segmented by means of marks in rt_info in corresponding tensors, each segment represent a single
    for(size_t i = 0; i < bindings.size(); ++i) {
        const auto& st = bindings[i].structural_type;
        const auto& indices = bindings[i].inputs;
        if(st.is<StructuralType::Tensor>()) {
            auto tensor = st.as<StructuralType::Tensor>();
            const Any& element_type = tensor.element_type;
            if(element_type.is<element::Type>()) {
                assert(indices.size() == 1);
                result.push_back(std::make_shared<Tensor<const descriptor::Tensor*>>(&node_inputs[indices[0]].get_tensor()));
            } else if(element_type.is<StructuralType::Str>()) {
                if(indices.size() == 1) {
                    result.push_back(std::make_shared<TensorStr<const descriptor::Tensor*>>(&node_inputs[indices[0]].get_tensor()));
                } else {
                    result.push_back(std::make_shared<TensorStr<const descriptor::Tensor*>>(&node_inputs[indices[0]].get_tensor(), &node_inputs[indices[1]].get_tensor(), &node_inputs[indices[2]].get_tensor()));
                }
            }
        } else {
            throw std::string("Type binding has unsupported structural data type");
        }
    }
}


class TENSORFLOW_API StructuralTypeMapAttribute : public ov::RuntimeAttribute {
public:
    OPENVINO_RTTI("structural_type_mapX", "0");

    StructuralTypeMapAttribute() = default;

    StructuralTypeMapAttribute(const BindInputs& value) : value(value) {}

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        // TODO: Implement deserialization; now only serialization works
        auto str_value = to_string();
        visitor.on_attribute("value", str_value);
        return true;
    }

    std::string to_string() const override {
        std::ostringstream buf;

        for(size_t i = 0; i < value.size(); ++i) {
            if(i > 0) {
                buf << ", ";
            }
            StructuralType::print(buf, value[i].structural_type);
            const auto& inputs = value[i].inputs;
            buf << '(';
            for(size_t j = 0; j < inputs.size(); ++j) {
                if(j > 0) {
                    buf << ", ";
                }
                buf << inputs[j];
            }
            buf << ')';
        }

        return buf.str();
    }

    void set_input (Node::RTMap& rt_info) {
        rt_info["structural_type_input_map"] = *this;
    }

    void set_output (Node::RTMap& rt_info) {
        rt_info["structural_type_output_map"] = *this;
    }

    static BindInputs get_input (const Node::RTMap& rt_info) {
        auto p = rt_info.find("structural_type_input_map");
        if(p != rt_info.end()) {
            return p->second.as<StructuralTypeMapAttribute>().value;
        } else {
            return BindInputs();
        }
    }

    static BindInputs get_output (const Node::RTMap& rt_info) {
        auto p = rt_info.find("structural_type_output_map");
        if(p != rt_info.end()) {
            return p->second.as<StructuralTypeMapAttribute>().value;
        } else {
            return BindInputs();
        }
    }

    BindInputs value;
};


}
}
}
}
