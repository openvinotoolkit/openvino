// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include <transformations_visibility.hpp>


#include "ngraph/op/op.hpp"
#include "ngraph/strides.hpp"
#include "ngraph/partial_shape.hpp"

namespace ngraph {
namespace op {

class TRANSFORMATIONS_API FrameworkNodeAttrs {
public:
    using attrs_t = std::unordered_map<std::string, std::string>;

    void set_opset_name(const std::string& opset_name) { m_opset_name = opset_name; }

    void set_type_name(const std::string& type_name) { m_type_name = type_name; }

    const std::string& get_opset_name() const { return m_opset_name; }

    const std::string& get_type_name() const { return m_type_name; }

    attrs_t::iterator begin() { return m_attrs.begin(); }

    attrs_t::iterator end() { return m_attrs.end(); }

    attrs_t::const_iterator begin() const { return m_attrs.begin(); }

    attrs_t::const_iterator end() const { return m_attrs.end(); }

    std::string operator[](const std::string & key) { return m_attrs[key]; }

    std::string at(const std::string & key) const { return m_attrs.at(key); }

    bool operator== (const FrameworkNodeAttrs & other) const {
        return m_type_name == other.m_type_name && m_opset_name == other.m_opset_name && m_attrs == m_attrs;
    }

private:
    std::string m_type_name;
    std::string m_opset_name;

    std::unordered_map<std::string, std::string> m_attrs;
};

class TRANSFORMATIONS_API FrameworkNode : public Op {
public:
    NGRAPH_RTTI_DECLARATION;

    explicit FrameworkNode(const OutputVector& inputs);

    void validate_and_infer_types() override;

    bool visit_attributes(AttributeVisitor& visitor) override {
        visitor.on_attribute("framework_node_attrs", m_attrs);
        return true;
    }

    const FrameworkNodeAttrs & get_attrs() const { return m_attrs; }

    void set_attrs(const FrameworkNodeAttrs & attrs) { m_attrs = attrs; }

    std::shared_ptr<Node>
        clone_with_new_inputs(const OutputVector& new_args) const override;

private:
    std::vector<std::tuple<ngraph::PartialShape, ngraph::element::Type>> m_inputs_desc;

    FrameworkNodeAttrs m_attrs;
};
} // namespace op

template <>
class TRANSFORMATIONS_API AttributeAdapter<op::FrameworkNodeAttrs>
    : public DirectValueAccessor<op::FrameworkNodeAttrs> {
public:
    AttributeAdapter(op::FrameworkNodeAttrs& value);

    static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<FrameworkNodeAttr>", 0};
    const DiscreteTypeInfo& get_type_info() const override { return type_info; }
};
} // namespace ngraph
