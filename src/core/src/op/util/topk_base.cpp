// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/topk_base.hpp"

#include <limits>

#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/util/precision_sensitive_attribute.hpp"
#include "topk_shape_inference.hpp"

namespace {
constexpr auto UNKNOWN_NORMALIZED_AXIS = std::numeric_limits<uint64_t>::max();
}

ov::op::util::TopKBase::TopKBase(const Output<Node>& data,
                                 const Output<Node>& k,
                                 const int64_t axis,
                                 const std::string& mode,
                                 const std::string& sort,
                                 const element::Type& index_element_type)
    : TopKBase(data, k, axis, as_enum<TopKMode>(mode), as_enum<TopKSortType>(sort), index_element_type) {}

ov::op::util::TopKBase::TopKBase(const Output<Node>& data,
                                 const Output<Node>& k,
                                 const int64_t axis,
                                 const TopKMode mode,
                                 const TopKSortType sort,
                                 const element::Type& index_element_type)
    : Op{{data, k}},
      m_axis{axis},
      m_normalized_axis{UNKNOWN_NORMALIZED_AXIS},
      m_mode{mode},
      m_sort{sort},
      m_index_element_type{index_element_type} {
    ov::mark_as_precision_sensitive(input(1));
}

void ov::op::util::TopKBase::validate_and_infer_types() {
    OV_OP_SCOPE(util_TopK_Base_validate_and_infer_types);

    k_type_check(get_input_element_type(1));

    set_axis(get_input_partial_shape(0).rank(), get_provided_axis());

    const auto output_shapes = shape_infer(this, ov::util::get_node_input_partial_shapes(*this));

    set_output_type(0, get_input_element_type(0), output_shapes[0]);
    set_output_type(1, m_index_element_type, output_shapes[1]);
}

bool ov::op::util::TopKBase::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(util_TopK_Base_visit_attributes);
    visitor.on_attribute("axis", m_axis);
    visitor.on_attribute("mode", m_mode);
    visitor.on_attribute("sort", m_sort);
    visitor.on_attribute("index_element_type", m_index_element_type);
    return true;
}

void ov::op::util::TopKBase::k_type_check(const element::Type& k_element_type) const {
    NODE_VALIDATION_CHECK(this,
                          k_element_type.is_integral_number(),
                          "K input has to be an integer type, which does match the provided one:",
                          k_element_type);
}

size_t ov::op::util::TopKBase::read_k_from_constant_node(const std::shared_ptr<Node>& node,
                                                         const element::Type& k_element_type) const {
    k_type_check(k_element_type);

    const auto k_constant = ov::as_type_ptr<op::v0::Constant>(node);

    size_t k = 0;

    switch (static_cast<element::Type_t>(k_element_type)) {
    case element::Type_t::i8:
        k = validate_and_get_k<int8_t>(k_constant);
        break;
    case element::Type_t::i16:
        k = validate_and_get_k<int16_t>(k_constant);
        break;
    case element::Type_t::i32:
        k = validate_and_get_k<int32_t>(k_constant);
        break;
    case element::Type_t::i64:
        k = validate_and_get_k<int64_t>(k_constant);
        break;
    case element::Type_t::u8:
        k = validate_and_get_k<uint8_t>(k_constant);
        break;
    case element::Type_t::u16:
        k = validate_and_get_k<uint16_t>(k_constant);
        break;
    case element::Type_t::u32:
        k = validate_and_get_k<uint32_t>(k_constant);
        break;
    case element::Type_t::u64:
        k = validate_and_get_k<uint64_t>(k_constant);
        break;
    default:
        break;
    }

    return k;
}

template <typename T>
size_t ov::op::util::TopKBase::validate_and_get_k(const std::shared_ptr<op::v0::Constant>& k_constant) const {
    const auto k_const_contents = k_constant->get_vector<T>();

    NODE_VALIDATION_CHECK(this,
                          k_const_contents.size() == 1,
                          "Only one value (scalar) should be provided as the 'K' input to TopK",
                          " (got ",
                          k_const_contents.size(),
                          " elements).");

    NODE_VALIDATION_CHECK(this,
                          k_const_contents[0] >= 0,
                          "The value of 'K' must be more or equal zero.",
                          " (got ",
                          k_const_contents[0],
                          ").");

    return static_cast<size_t>(k_const_contents[0]);
}

void ov::op::util::TopKBase::set_k(size_t k) {
    this->input(1).replace_source_output(op::v0::Constant::create(element::i64, ov::Shape{}, {k})->output(0));
}

size_t ov::op::util::TopKBase::get_k() const {
    size_t k = 0;
    if (op::util::is_constant(input_value(1).get_node())) {
        k = read_k_from_constant_node(input_value(1).get_node_shared_ptr(), get_input_element_type(1));
    }

    if (k == 0 && get_input_partial_shape(0).is_static()) {
        k = get_input_partial_shape(0).to_shape()[m_normalized_axis];
    }
    return k;
}

void ov::op::util::TopKBase::set_axis(const int64_t axis) {
    set_axis(get_input_partial_shape(0).rank(), axis);
}

void ov::op::util::TopKBase::set_axis(const Rank& input_rank, const int64_t axis) {
    m_normalized_axis =
        input_rank.is_static() ? ov::util::try_normalize_axis(axis, input_rank, *this) : UNKNOWN_NORMALIZED_AXIS;
    m_axis = axis;
}

uint64_t ov::op::util::TopKBase::get_axis() const {
    NODE_VALIDATION_CHECK(this, m_normalized_axis != UNKNOWN_NORMALIZED_AXIS, "Normalized axis of TopK is unknown");

    return m_normalized_axis;
}
