// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "openvino/core/partial_shape.hpp"
#include "openvino/core/strides.hpp"
#include "openvino/core/type/non_tensor_type.hpp"   // TODO: Required?
#include "openvino/op/util/multi_subgraph_base.hpp"

namespace ov {
namespace op {
namespace util {

// TODO: Required?
class OPENVINO_API StructuralTypeAttribute : public ov::RuntimeAttribute {
public:
    OPENVINO_RTTI("structural_type", "0");

    StructuralTypeAttribute() = default;

    StructuralTypeAttribute(const ov::Any& value) : value(value) {}

    //Any merge(const ngraph::NodeVector& nodes) const override;

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        // TODO: Implement deserialization; now only serialization works
        auto str_value = to_string();
        visitor.on_attribute("value", str_value);
        return true;
    }

    std::string to_string() const override {
        std::ostringstream str;
        ov::element::StructuralType::print(str, value);
        return str.str();
    }

    ov::Any value;

    static void copy (const Node::RTMap& src, Node::RTMap& dst);
    static bool has_type (const Node::RTMap& src, const ov::Any& type);
    static void move_to_original (Node::RTMap& rt_info);
    static ov::Any get (const Node::RTMap& src);
};

class OPENVINO_API StructPack : public ov::op::Op {
public:
    OPENVINO_OP("INTERNAL::StructPack");

    StructPack(const OutputVector& arguments, Any res_type, const PartialShape& res_shape, element::Type_t element_type = element::dynamic)
        : ov::op::Op(arguments), m_res_type(res_type), m_res_shape(res_shape), m_element_type(element_type) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        set_output_type(0, m_element_type, m_res_shape);
        // TODO: Requiered?
        get_output_tensor(0).get_rt_info()["structural_type"] = StructuralTypeAttribute(m_res_type);
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const OutputVector& inputs) const override {
        return std::make_shared<StructPack>(inputs, m_res_type, m_res_shape, m_element_type);
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        // FIXME: Serialization for dubug purposes only, there is no deserialization.
        std::string m_res_type_str =m_res_type.as<std::string>();
        visitor.on_attribute("res_type", m_res_type_str);
        visitor.on_attribute("res_shape", m_res_shape);
        return true;
    }

    bool has_evaluate() const {
        return false;
    }

    Any m_res_type;
    PartialShape m_res_shape;
    element::Type_t m_element_type;
};

}  // namespace util
}  // namespace op
}  // namespace ov
