// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/pass/pattern/op/op.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"

namespace ov::pass::pattern {
namespace op {
class OPENVINO_API WrapType : public Pattern {
public:
    OPENVINO_RTTI("WrapType");

    explicit WrapType(const std::vector<NodeTypeInfo>& wrapped_types) : Pattern(), m_wrapped_types(wrapped_types) {
        set_output_type(0, element::Type_t::dynamic, PartialShape::dynamic());
    }

    template <typename TPredicate>
    WrapType(const std::vector<NodeTypeInfo>& wrapped_types,
             const TPredicate& pred,
             const OutputVector& input_values = {})
        : Pattern(input_values, Predicate(pred)),
          m_wrapped_types(wrapped_types) {
        set_output_type(0, element::Type_t::dynamic, PartialShape::dynamic());
    }

    explicit WrapType(NodeTypeInfo wrapped_type) : Pattern(), m_wrapped_types({wrapped_type}) {
        set_output_type(0, element::Type_t::dynamic, PartialShape::dynamic());
    }

    template <typename TPredicate>
    WrapType(NodeTypeInfo wrapped_type, const TPredicate& pred, const OutputVector& input_values = {})
        : Pattern(input_values, Predicate(pred)),
          m_wrapped_types({wrapped_type}) {
        set_output_type(0, element::Type_t::dynamic, PartialShape::dynamic());
    }

    bool match_value(pattern::Matcher* matcher,
                     const Output<Node>& pattern_value,
                     const Output<Node>& graph_value) override;

    NodeTypeInfo get_wrapped_type() const;

    const std::vector<NodeTypeInfo>& get_wrapped_types() const;
    std::ostream& write_type_description(std::ostream& out) const override;
    
    // Track when multiple outputs are expected
    void set_wrapped_output_size(size_t size) {
        set_output_size(size);
        m_explicit_output_size = size;
    }

private:
    std::vector<NodeTypeInfo> m_wrapped_types;
    size_t m_explicit_output_size = 0;  // 0 means not explicitly set
};
}  // namespace op

template <class T>
void collect_wrap_info(std::vector<DiscreteTypeInfo>& info) {
    info.emplace_back(T::get_type_info_static());
}

template <class T, class... Targs, typename std::enable_if<sizeof...(Targs) != 0, bool>::type = true>
void collect_wrap_info(std::vector<DiscreteTypeInfo>& info) {
    collect_wrap_info<T>(info);
    collect_wrap_info<Targs...>(info);
}

template <class... Args,
          typename TPredicate,
          typename std::enable_if_t<std::is_constructible_v<op::Predicate, TPredicate>>* = nullptr>
std::shared_ptr<Node> wrap_type(const PatternOps& inputs, const TPredicate& pred, const Attributes& attrs = {}) {
    std::vector<DiscreteTypeInfo> info;
    collect_wrap_info<Args...>(info);
    return std::make_shared<op::WrapType>(
        info,
        (attrs.empty() ? op::Predicate(pred) : attrs_match(attrs) && op::Predicate(pred)),
        ov::OutputVector(inputs));
}

template <class... Args,
          typename TPredicate,
          typename std::enable_if_t<std::is_constructible_v<op::Predicate, TPredicate> &&
                                    !std::is_constructible_v<PatternOps, TPredicate>>* = nullptr>
std::shared_ptr<Node> wrap_type(const TPredicate& pred, const Attributes& attrs = {}) {
    return wrap_type<Args...>(PatternOps{}, op::Predicate(pred), attrs);
}

template <class... Args>
std::shared_ptr<Node> wrap_type(const PatternOps& inputs = {}, const Attributes& attrs = {}) {
    return wrap_type<Args...>(inputs, (attrs.empty() ? op::Predicate() : attrs_match(attrs)));
}

template <class... Args>
std::shared_ptr<Node> wrap_type(std::initializer_list<std::pair<const std::string, ov::Any>>&& attrs) {
    return wrap_type<Args...>(PatternOps{}, Attributes(attrs));
}

OPENVINO_API std::shared_ptr<Node> wrap_const();
}  // namespace ov::pass::pattern
