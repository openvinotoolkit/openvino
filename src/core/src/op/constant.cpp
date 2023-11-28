// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/constant.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <sstream>

#include "compare.hpp"
#include "element_visitor.hpp"
#include "itt.hpp"
#include "ngraph/runtime/aligned_buffer.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/core/type/nf4.hpp"
#include "openvino/reference/utils/type_util.hpp"
#include "openvino/runtime/shared_buffer.hpp"

namespace ov {
namespace op {

template <class TContainer>
TContainer convert_values_to(std::vector<int64_t>&& values, const Shape& shape) {
    auto out = TContainer(shape_size(shape));
    std::replace_copy_if(values.begin(), values.end(), out.begin(), cmp::Less<int64_t>(0), 0);
    return out;
}

namespace {
template <typename T, typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
std::string to_cpp_string(T value) {
    if (std::isnan(value)) {
        return "NAN";
    } else if (std::isinf(value)) {
        return std::signbit(value) ? "-INFINITY" : "INFINITY";
    } else {
        std::stringstream ss;
        ss << value;
        return ss.str();
    }
}

std::vector<double> from_string_vector(const std::vector<std::string>& str_values) {
    std::vector<double> values;
    values.reserve(str_values.size());
    std::transform(str_values.cbegin(), str_values.cend(), std::back_inserter(values), [](const std::string& s) {
        size_t pos;
        auto v = std::stold(s, &pos);
        OPENVINO_ASSERT(s.size() == pos, "Could not parse literal '", s, "'");
        return v;
    });
    return values;
}
}  // namespace

namespace v0 {
OPENVINO_SUPPRESS_DEPRECATED_START
std::shared_ptr<AlignedBuffer> Constant::legacy_to_ov_aligned_buffer(
    const std::shared_ptr<ngraph::runtime::AlignedBuffer>& buffer) {
    return std::make_shared<SharedBuffer<std::shared_ptr<ngraph::runtime::AlignedBuffer>>>(buffer->get_ptr<char>(),
                                                                                           buffer->size(),
                                                                                           buffer);
}

Constant::Constant(const std::shared_ptr<ngraph::runtime::Tensor>& tensor) {
    m_element_type = tensor->get_element_type();
    m_shape = tensor->get_shape();
    // Share data from HostTensor if we work with it
    // And copy data in other cas
    if (auto hostTensor = std::dynamic_pointer_cast<ngraph::runtime::HostTensor>(tensor)) {
        m_data = std::make_shared<SharedBuffer<std::shared_ptr<ngraph::runtime::Tensor>>>(
            static_cast<char*>(hostTensor->get_data_ptr()),
            tensor->get_size_in_bytes(),
            tensor);
    } else {
        constructor_validate_and_infer_types();
        allocate_buffer(false);
        tensor->read(get_data_ptr_nc(), tensor->get_size_in_bytes());
        if (m_element_type == ov::element::string) {
            // we need to re-initialize memory with separate (newly created) std::string objects with the same values
            auto size = shape_size(m_shape);
            auto string_ptr = static_cast<std::string*>(get_data_ptr_nc());
            std::transform(string_ptr, string_ptr + size, string_ptr, [](std::string value) {
                return value;
            });
        }
    }
    constructor_validate_and_infer_types();
}
OPENVINO_SUPPRESS_DEPRECATED_END

Constant::Constant(const Tensor& tensor)
    : m_element_type{tensor.get_element_type()},
      m_shape{tensor.get_shape()},
      m_data{
          std::make_shared<SharedBuffer<Tensor>>(static_cast<char*>(tensor.data()), tensor.get_byte_size(), tensor)} {
    constructor_validate_and_infer_types();
}

Constant::Constant(const element::Type& type, const Shape& shape, const std::vector<std::string>& values)
    : Constant(false, type, shape) {
    NODE_VALIDATION_CHECK(this,
                          values.size() == 1 || values.size() == shape_size(m_shape),
                          "Did not get the expected number of literals for a constant of shape ",
                          m_shape,
                          " (got ",
                          values.size(),
                          ", expected ",
                          (shape_size(m_shape) == 1 ? "" : "1 or "),
                          shape_size(m_shape),
                          ").");

    if (type == element::string) {
        if (values.size() == 1) {
            fill_data(type, values.front());
        } else {
            write_values(values);
        }
    } else {
        auto parsed_values = from_string_vector(values);
        if (parsed_values.size() == 1) {
            fill_data(type, parsed_values.front());
        } else {
            write_values(parsed_values);
        }
    }
    const auto is_checked_and_identical = (values.size() == 1) && (shape_size(m_shape) != 1);
    update_identical_flags(is_checked_and_identical, is_checked_and_identical);
}

Constant::Constant(const element::Type& type, const Shape& shape) : Constant(true, type, shape) {}

Constant::Constant(bool memset_allocation, const element::Type& type, const Shape& shape)
    : m_element_type(type),
      m_shape(shape) {
    allocate_buffer(memset_allocation);
    constructor_validate_and_infer_types();
}

void Constant::allocate_buffer(bool memset_allocation) {
    // memset_allocation flag is to switch on initialization of objects in memory for element::string type
    // and set memory to zero for numeric element types
    m_data = std::make_shared<AlignedBuffer>(mem_size(), host_alignment());
    if (memset_allocation) {
        if (m_element_type == ov::element::string) {
            // initialize std::string objects in memory
            auto size = shape_size(m_shape);
            auto string_ptr = static_cast<std::string*>(get_data_ptr_nc());
            std::uninitialized_fill_n(string_ptr, size, std::string());
        } else {
            std::memset(m_data->get_ptr(), 0, m_data->size());
        }
    }
}

Constant::Constant(const element::Type& type, const Shape& shape, const void* data) : Constant(false, type, shape) {
    if (m_element_type == ov::element::string) {
        auto num_elements = shape_size(m_shape);
        const std::string* src_strings = static_cast<const std::string*>(data);
        std::string* dst_strings = static_cast<std::string*>(get_data_ptr_nc());
        std::uninitialized_copy_n(src_strings, num_elements, dst_strings);
    } else {
        std::memcpy(get_data_ptr_nc(), data, mem_size());
    }
}

Constant::Constant(const Constant& other)
    : m_element_type{other.m_element_type},
      m_shape{other.m_shape},
      m_data{other.m_data},
      m_all_elements_bitwise_identical{other.m_all_elements_bitwise_identical.load()},
      m_all_elements_bitwise_identical_checked{other.m_all_elements_bitwise_identical_checked.load()} {
    constructor_validate_and_infer_types();
}

Constant::Constant(const Constant& other, const Shape& new_shape)
    : m_element_type{other.m_element_type},
      m_shape{new_shape},
      m_data{other.m_data},
      m_all_elements_bitwise_identical{other.m_all_elements_bitwise_identical.load()},
      m_all_elements_bitwise_identical_checked{other.m_all_elements_bitwise_identical_checked.load()} {
    const auto new_size = shape_size(new_shape);
    const auto other_size = shape_size(other.m_shape);
    OPENVINO_ASSERT(other_size == new_size, "ov::Shape size ", new_size, " is not equal to ", other_size);
    constructor_validate_and_infer_types();
}

Constant::~Constant() = default;

struct ValueToString : ov::element::NotSupported<std::string> {
    using ov::element::NotSupported<std::string>::visit;

    template <ov::element::Type_t ET, typename std::enable_if<ET == element::f64>::type* = nullptr>
    static result_type visit(const Constant* const c, const size_t index) {
        return to_cpp_string(c->get_element_value<ET>(index));
    }

    template <ov::element::Type_t ET,
              typename std::enable_if<ov::is_floating_point<fundamental_type_for<ET>>() && ET != element::f64>::type* =
                  nullptr>
    static result_type visit(const Constant* const c, const size_t index) {
        return to_cpp_string<float>(c->get_element_value<ET>(index));
    }

    template <ov::element::Type_t ET,
              typename std::enable_if<std::is_integral<ov::fundamental_type_for<ET>>::value>::type* = nullptr>
    static result_type visit(const Constant* const c, const size_t index) {
        return std::to_string(c->get_element_value<ET>(index));
    }

    template <ov::element::Type_t ET,
              typename std::enable_if<std::is_same<fundamental_type_for<ET>, std::string>::value>::type* = nullptr>
    static result_type visit(const Constant* const c, const size_t index) {
        return c->get_element_value<ET>(index);
    }
};

std::string Constant::convert_value_to_string(size_t index) const {
    using namespace ov::element;
    return IfTypeOf<boolean, bf16, f16, f32, f64, i4, i8, i16, i32, i64, u1, u4, u8, u16, u32, u64, nf4, string>::apply<
        ValueToString>(get_element_type(), this, index);
}

size_t Constant::get_byte_size() const {
    // Returns 0 when shape is "empty" (equals 0).
    // TODO: refactor shape_size(m_shape) calculations and store it as a member.
    return shape_size(m_shape) ? m_data->size() : 0;
}

const void* Constant::get_data_ptr() const {
    return (m_data ? m_data->get_ptr() : nullptr);
}

void* Constant::get_data_ptr_nc() {
    return (m_data ? m_data->get_ptr() : nullptr);
}

struct ValuesToString : ov::element::NotSupported<void> {
    using ov::element::NotSupported<void>::visit;

    template <ov::element::Type_t ET,
              class T = fundamental_type_for<ET>,
              typename std::enable_if<ov::is_floating_point<T>()>::type* = nullptr>
    static result_type visit(const Constant* const c, std::vector<std::string>& strs) {
        for (auto&& v : c->get_vector<T>()) {
            strs.push_back(to_cpp_string<double>(v));
        }
    }

    template <ov::element::Type_t ET,
              class T = fundamental_type_for<ET>,
              typename std::enable_if<std::is_integral<T>::value && !std::is_same<T, int8_t>::value>::type* = nullptr>
    static result_type visit(const Constant* const c, std::vector<std::string>& strs) {
        for (auto&& v : c->get_vector<T>()) {
            strs.push_back(std::to_string(v));
        }
    }

    template <ov::element::Type_t ET,
              typename std::enable_if<std::is_same<fundamental_type_for<ET>, int8_t>::value>::type* = nullptr>
    static result_type visit(const Constant* const c, std::vector<std::string>& strs) {
        for (auto&& v : c->cast_vector<int8_t>()) {
            strs.push_back(std::to_string(v));
        }
    }

    template <ov::element::Type_t ET,
              typename std::enable_if<std::is_same<fundamental_type_for<ET>, std::string>::value>::type* = nullptr>
    static result_type visit(const Constant* const c, std::vector<std::string>& strs) {
        strs = c->cast_vector<std::string>();
    }
};

std::vector<std::string> Constant::get_value_strings() const {
    std::vector<std::string> out;
    using namespace ov::element;
    IfTypeOf<boolean, bf16, f16, f32, f64, i4, i8, i16, i32, i64, u1, u4, u8, u16, u32, u64, nf4, string>::apply<
        ValuesToString>(get_element_type(), this, out);
    return out;
}

Shape Constant::get_shape_val() const {
    OPENVINO_ASSERT(m_element_type.is_integral_number());
    return convert_values_to<Shape>(cast_vector<int64_t>(), m_shape);
}

Strides Constant::get_strides_val() const {
    OPENVINO_ASSERT(m_element_type == element::i64);
    return convert_values_to<Strides>(get_vector<int64_t>(), m_shape);
}

Coordinate Constant::get_coordinate_val() const {
    OPENVINO_ASSERT(m_element_type == element::i64);
    return convert_values_to<Coordinate>(get_vector<int64_t>(), m_shape);
}

CoordinateDiff Constant::get_coordinate_diff_val() const {
    OPENVINO_ASSERT(m_element_type == element::i64);
    return convert_values_to<CoordinateDiff>(get_vector<int64_t>(), m_shape);
}

AxisVector Constant::get_axis_vector_val() const {
    OPENVINO_ASSERT(m_element_type.is_integral_number());
    return convert_values_to<AxisVector>(cast_vector<int64_t>(), m_shape);
}

AxisSet Constant::get_axis_set_val() const {
    OPENVINO_ASSERT(m_element_type.is_integral_number());
    const auto values = cast_vector<int64_t>();
    AxisSet out;
    std::replace_copy_if(values.begin(), values.end(), std::inserter(out, out.end()), cmp::Less<int64_t>(0), 0);
    return out;
}

std::shared_ptr<Node> Constant::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_Constant_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<Constant>(*this);
}

template <typename T>
bool test_bitwise_identical(const T* data, const size_t size) {
    OPENVINO_ASSERT(size == 0 || data != nullptr);
    return std::all_of(data, data + size, [&](const T value) {
        return value == data[0];
    });
}

bool Constant::are_all_data_elements_bitwise_identical() const {
    bool all_identical;

    switch (m_element_type) {
    case element::Type_t::boolean:
    case element::Type_t::i8:
    case element::Type_t::u8:
        all_identical = test_bitwise_identical(get_data_ptr<uint8_t>(), shape_size(m_shape));
        break;
    case element::Type_t::bf16:
    case element::Type_t::f16:
    case element::Type_t::i16:
    case element::Type_t::u16:
        all_identical = test_bitwise_identical(get_data_ptr<uint16_t>(), shape_size(m_shape));
        break;
    case element::Type_t::f32:
    case element::Type_t::i32:
    case element::Type_t::u32:
        all_identical = test_bitwise_identical(get_data_ptr<uint32_t>(), shape_size(m_shape));
        break;
    case element::Type_t::f64:
    case element::Type_t::i64:
    case element::Type_t::u64:
        all_identical = test_bitwise_identical(get_data_ptr<uint64_t>(), shape_size(m_shape));
        break;
    case element::Type_t::string:
        all_identical = test_bitwise_identical(get_data_ptr<std::string>(), shape_size(m_shape));
        break;
    default:
        all_identical = false;
        break;
    }
    return all_identical;
}

void Constant::update_identical_flags(bool is_checked, bool identical_value) const {
    m_all_elements_bitwise_identical_checked = is_checked;
    m_all_elements_bitwise_identical = identical_value;
}

bool Constant::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_Constant_visit_attributes);
    const auto prev_shape = m_shape;
    const auto prev_type = m_element_type;
    visitor.on_attribute("element_type", m_element_type);
    visitor.on_attribute("shape", m_shape);

    const auto need_to_reallocate = (m_shape != prev_shape) || (prev_type != m_element_type);
    if (m_alloc_buffer_on_visit_attributes && need_to_reallocate) {
        // Filling in a fresh constant
        allocate_buffer(false);
    }
    visitor.on_attribute("value", m_data);
    update_identical_flags(false, false);
    return true;
}

bool Constant::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v0_Constant_evaluate);
    if (outputs.empty())
        outputs.emplace_back(m_element_type, m_shape);
    else
        outputs[0].set_shape(m_shape);
    if (m_element_type == ov::element::string) {
        auto num_elements = shape_size(m_shape);
        const std::string* src_strings = static_cast<const std::string*>(get_data_ptr());
        std::string* dst_strings = static_cast<std::string*>(outputs[0].data());
        std::copy_n(src_strings, num_elements, dst_strings);
    } else {
        std::memcpy(outputs[0].data(), get_data_ptr(), outputs[0].get_byte_size());
    }

    return true;
}

bool Constant::has_evaluate() const {
    OV_OP_SCOPE(v0_Constant_has_evaluate);
    return true;
}

bool Constant::evaluate_lower(TensorVector& outputs) const {
    return evaluate(outputs, {});
}
bool Constant::evaluate_upper(TensorVector& outputs) const {
    return evaluate(outputs, {});
}

uint8_t Constant::quantize_nf4(float x) {
    return ConvertNF4::quantize(x);
}
}  // namespace v0
}  // namespace op
}  // namespace ov
