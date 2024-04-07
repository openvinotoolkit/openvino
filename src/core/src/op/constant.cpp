// Copyright (C) 2018-2024 Intel Corporation
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
#include "openvino/core/type/element_iterator.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/core/type/nf4.hpp"
#include "openvino/reference/convert.hpp"
#include "openvino/reference/utils/type_util.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "openvino/runtime/string_aligned_buffer.hpp"
#include "openvino/runtime/tensor.hpp"

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

template <class T, typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
T str_to_value(const std::string& s, size_t* pos) {
    return static_cast<T>(std::is_signed<T>::value ? std::stoll(s, pos) : std::stoull(s, pos));
}

template <class T, typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
T str_to_value(const std::string& s, size_t* pos) {
    return static_cast<T>(std::stod(s, pos));
}

template <class T>
std::vector<T> from_string_vector(const std::vector<std::string>& str_values) {
    std::vector<T> values;
    values.reserve(str_values.size());
    std::transform(str_values.cbegin(), str_values.cend(), std::back_inserter(values), [](const std::string& s) {
        size_t pos;
        auto v = str_to_value<T>(s, &pos);
        OPENVINO_ASSERT(s.size() == pos, "Could not parse literal '", s, "'");
        return v;
    });
    return values;
}

template <element::Type_t ET, class U, typename std::enable_if<ET == element::u1>::type* = nullptr>
fundamental_type_for<ET> convert_if_in_element_range(const U& value) {
    using T = fundamental_type_for<ET>;
    return static_cast<T>(static_cast<bool>(value));
}

template <element::Type_t ET,
          class U,
          typename std::enable_if<ET == element::nf4 && std::is_integral<U>::value>::type* = nullptr>
fundamental_type_for<ET> convert_if_in_element_range(const U& value) {
    using T = fundamental_type_for<ET>;
    auto result = static_cast<T>(value);
    OPENVINO_ASSERT(0 <= result && result <= 15, "assigned value out of range u4 values");
    return result;
}

template <element::Type_t ET,
          class U,
          typename std::enable_if<ET == element::nf4 && !std::is_integral<U>::value>::type* = nullptr>
float convert_if_in_element_range(const U& value) {
    return static_cast<float>(value);
}

template <element::Type_t ET, class U, typename std::enable_if<ET == element::u4>::type* = nullptr>
fundamental_type_for<ET> convert_if_in_element_range(const U& value) {
    using T = fundamental_type_for<ET>;
    auto result = static_cast<T>(value);
    OPENVINO_ASSERT(0 <= result && result <= 15, "assigned value out of range u4 values");
    return result;
}

template <element::Type_t ET, class U, typename std::enable_if<ET == element::i4>::type* = nullptr>
fundamental_type_for<ET> convert_if_in_element_range(const U& value) {
    using T = fundamental_type_for<ET>;
    auto result = static_cast<T>(value);
    OPENVINO_ASSERT(-8 <= result && result <= 7, "assigned value out of range i4 values");
    return result;
}

template <element::Type_t ET, class T>
void fill_buffer(void* buffer, const Shape& shape, const T& value) {
    std::fill_n(element::iterator<ET>(buffer), shape_size(shape), convert_if_in_element_range<ET>(value));
}

template <element::Type_t ET, class U>
void cast_buffer(const void* buffer, size_t num_elements, std::vector<U>& output) {
    const auto first = element::iterator<ET>(buffer);
    using StorageType = fundamental_type_for<ET>;

    std::transform(first, first + num_elements, std::back_inserter(output), reference::detail::convert<StorageType, U>);
}

template <element::Type_t ET, class T>
void write_buffer(const std::vector<T>& source, void* buffer) {
    std::transform(source.begin(), source.end(), element::iterator<ET>(buffer), convert_if_in_element_range<ET, T>);
}
}  // namespace

namespace v0 {

Constant::Constant(const Tensor& tensor)
    : m_element_type{tensor.get_element_type()},
      m_shape{tensor.get_shape()},
      m_data{
          std::make_shared<SharedBuffer<Tensor>>(static_cast<char*>(tensor.data()), tensor.get_byte_size(), tensor)} {
    constructor_validate_and_infer_types();
}

Constant::Constant(const element::Type& type, const Shape& shape, const std::vector<std::string>& values)
    : Constant(false, type, shape) {
    const auto this_shape_size = shape_size(m_shape);
    const auto values_size = values.size();
    const auto has_single_value = (values_size == 1);
    NODE_VALIDATION_CHECK(this,
                          has_single_value || values_size == this_shape_size,
                          "Did not get the expected number of literals for a constant of shape ",
                          m_shape,
                          " (got ",
                          values_size,
                          ", expected ",
                          (this_shape_size == 1 ? "" : "1 or "),
                          this_shape_size,
                          ").");
    const auto is_checked_and_identical = has_single_value && (this_shape_size != 1);

    if (type == element::string) {
        fill_or_write(is_checked_and_identical, type, values);
    } else if (type.is_real()) {
        fill_or_write(is_checked_and_identical, type, from_string_vector<double>(values));
    } else if (type.is_signed()) {
        fill_or_write(is_checked_and_identical, type, from_string_vector<int64_t>(values));
    } else {
        fill_or_write(is_checked_and_identical, type, from_string_vector<uint64_t>(values));
    }
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
    const auto num_elements = shape_size(m_shape);
    const auto byte_size = element::get_byte_size(m_element_type, num_elements);
    if (m_element_type == ov::element::string) {
        m_data = std::make_shared<StringAlignedBuffer>(num_elements, byte_size, host_alignment(), memset_allocation);
    } else {
        m_data = std::make_shared<AlignedBuffer>(byte_size, host_alignment());
        if (memset_allocation) {
            std::memset(m_data->get_ptr(), 0, m_data->size());
        }
    }
}

Constant::Constant(const element::Type& type, const Shape& shape, const void* data) : Constant(false, type, shape) {
    const auto num_elements = shape_size(m_shape);
    if (m_element_type == ov::element::string) {
        const auto src_strings = static_cast<const std::string*>(data);
        const auto dst_strings = static_cast<std::string*>(get_data_ptr_nc());
        std::uninitialized_copy_n(src_strings, num_elements, dst_strings);
    } else {
        std::memcpy(get_data_ptr_nc(), data, element::get_byte_size(m_element_type, num_elements));
    }
}

Constant::Constant(const element::Type& type, const Shape& shape, const std::shared_ptr<ov::AlignedBuffer>& data)
    : m_element_type(type),
      m_shape(shape),
      m_data{data} {
    constructor_validate_and_infer_types();
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
    static result_type visit(const void* const ptr, const size_t index) {
        return to_cpp_string(element::iterator<ET>(ptr)[index]);
    }

    template <ov::element::Type_t ET,
              typename std::enable_if<ov::is_floating_point<fundamental_type_for<ET>>() && ET != element::f64>::type* =
                  nullptr>
    static result_type visit(const void* const ptr, const size_t index) {
        const auto it = element::iterator<ET>(ptr);
        return element::is_byte_type(ET) ? to_cpp_string<float>(it[index]) : to_cpp_string<float>(*(it + index));
    }

    template <ov::element::Type_t ET,
              typename std::enable_if<std::is_integral<ov::fundamental_type_for<ET>>::value>::type* = nullptr>
    static result_type visit(const void* const ptr, const size_t index) {
        const auto it = element::iterator<ET>(ptr) + index;
        return std::to_string(static_cast<typename ov::fundamental_type_for<ET>>(*it));
    }

    template <ov::element::Type_t ET, typename std::enable_if<ET == element::string>::type* = nullptr>
    static result_type visit(const void* const ptr, const size_t index) {
        return element::iterator<ET>(ptr)[index];
    }
};

std::string Constant::convert_value_to_string(size_t index) const {
    using namespace ov::element;
    return IfTypeOf<boolean,
                    bf16,
                    f16,
                    f32,
                    f64,
                    i4,
                    i8,
                    i16,
                    i32,
                    i64,
                    u1,
                    u4,
                    u8,
                    u16,
                    u32,
                    u64,
                    nf4,
                    f8e4m3,
                    f8e5m2,
                    string>::apply<ValueToString>(get_element_type(), get_data_ptr(), index);
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
    static result_type visit(const void* const ptr, const size_t num_elements, std::vector<std::string>& strs) {
        const auto first = element::iterator<ET>(ptr);
        std::transform(first, first + num_elements, std::back_inserter(strs), to_cpp_string<double>);
    }

    template <ov::element::Type_t ET,
              class T = fundamental_type_for<ET>,
              typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
    static result_type visit(const void* const ptr, const size_t num_elements, std::vector<std::string>& strs) {
        const auto first = element::iterator<ET>(ptr);
        std::transform(first, first + num_elements, std::back_inserter(strs), [](const T v) {
            return std::to_string(v);
        });
    }

    template <ov::element::Type_t ET, typename std::enable_if<ET == element::string>::type* = nullptr>
    static result_type visit(const void* const ptr, const size_t num_elements, std::vector<std::string>& strs) {
        std::copy_n(element::iterator<ET>(ptr), num_elements, std::back_inserter(strs));
    }
};

std::vector<std::string> Constant::get_value_strings() const {
    std::vector<std::string> out;
    using namespace ov::element;
    IfTypeOf<boolean, bf16, f16, f32, f64, i4, i8, i16, i32, i64, u1, u4, u8, u16, u32, u64, nf4, string>::apply<
        ValuesToString>(get_element_type(), get_data_ptr(), shape_size(m_shape), out);
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

void Constant::validate_and_infer_types() {
    set_output_type(0, m_element_type, m_shape);
}

bool Constant::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_Constant_visit_attributes);
    const auto prev_shape = m_shape;
    const auto prev_type = m_element_type;
    visitor.on_attribute("element_type", m_element_type);
    visitor.on_attribute("shape", m_shape);

    const auto need_to_reallocate = (m_shape != prev_shape) || (prev_type != m_element_type);
    const auto is_string_constant = (m_element_type == element::string);
    if (m_alloc_buffer_on_visit_attributes && need_to_reallocate) {
        // string objects initialization is required, others filling in a fresh constant
        allocate_buffer(is_string_constant);
    }

    if (is_string_constant) {
        if (auto string_aligned_buffer = std::dynamic_pointer_cast<ov::StringAlignedBuffer>(m_data)) {
            visitor.on_attribute("value", string_aligned_buffer);
        } else if (auto shared_string_tensor = std::dynamic_pointer_cast<ov::SharedBuffer<ov::Tensor>>(m_data)) {
            auto shared_string_buffer =
                std::make_shared<ov::SharedStringAlignedBuffer>(shared_string_tensor->get_ptr<char>(),
                                                                shared_string_tensor->size());
            visitor.on_attribute("value", shared_string_buffer);
        } else {
            // deserialization case when buffer does not exist yet
            std::shared_ptr<ov::StringAlignedBuffer> string_aligned_buffer;
            visitor.on_attribute("value", string_aligned_buffer);
            m_data = string_aligned_buffer;
        }
    } else {
        visitor.on_attribute("value", m_data);
    }
    update_identical_flags(false, false);
    return true;
}

bool Constant::get_all_data_elements_bitwise_identical() const {
    if (!m_all_elements_bitwise_identical_checked) {
        update_identical_flags(true, are_all_data_elements_bitwise_identical());
    }
    return m_all_elements_bitwise_identical;
}

void Constant::alloc_buffer_on_visit_attributes(bool val) {
    m_alloc_buffer_on_visit_attributes = val;
}

bool Constant::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v0_Constant_evaluate);
    if (outputs.empty())
        outputs.emplace_back(m_element_type, m_shape);
    else
        outputs[0].set_shape(m_shape);

    if (m_element_type == ov::element::string) {
        auto num_elements = shape_size(m_shape);
        auto src_strings = static_cast<const std::string*>(get_data_ptr());
        auto dst_strings = static_cast<std::string*>(outputs[0].data());
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

bool Constant::constant_fold(OutputVector&, const OutputVector&) {
    return false;
}

template <>
Constant::LPBuffer<element::u1>::LPBuffer(void* ptr)
    : iter{std::make_shared<lp_iter>(reinterpret_cast<ov::fundamental_type_for<element::u1>*>(ptr))} {}

template <>
Constant::LPBuffer<element::u4>::LPBuffer(void* ptr)
    : iter{std::make_shared<lp_iter>(reinterpret_cast<ov::fundamental_type_for<element::u4>*>(ptr))} {}

template <>
Constant::LPBuffer<element::i4>::LPBuffer(void* ptr)
    : iter{std::make_shared<lp_iter>(reinterpret_cast<ov::fundamental_type_for<element::i4>*>(ptr))} {}

template <>
Constant::LPBuffer<element::nf4>::LPBuffer(void* ptr)
    : iter{std::make_shared<lp_iter>(reinterpret_cast<ov::fundamental_type_for<element::nf4>*>(ptr))} {}

template <>
void Constant::LPBuffer<element::u1>::write(const float value) {
    iter->operator*() = convert_if_in_element_range<element::u1>(value);
}

template <>
void Constant::LPBuffer<element::u4>::write(const float value) {
    iter->operator*() = convert_if_in_element_range<element::u4>(value);
}

template <>
void Constant::LPBuffer<element::i4>::write(const float value) {
    iter->operator*() = convert_if_in_element_range<element::i4>(value);
}

template <>
void Constant::LPBuffer<element::nf4>::write(const float value) {
    iter->operator*() = convert_if_in_element_range<element::nf4>(value);
}

template <>
ov::fundamental_type_for<element::u1> Constant::LPBuffer<element::u1>::read() const {
    return iter->operator*();
}

template <>
ov::fundamental_type_for<element::u4> Constant::LPBuffer<element::u4>::read() const {
    return iter->operator*();
}

template <>
ov::fundamental_type_for<element::i4> Constant::LPBuffer<element::i4>::read() const {
    return iter->operator*();
}

template <>
ov::fundamental_type_for<element::nf4> Constant::LPBuffer<element::nf4>::read() const {
    return iter->operator*();
}

template <>
Constant::LPBuffer<element::u1>& Constant::LPBuffer<element::u1>::operator++() {
    iter->operator++();
    return *this;
}

template <>
Constant::LPBuffer<element::u4>& Constant::LPBuffer<element::u4>::operator++() {
    iter->operator++();
    return *this;
}

template <>
Constant::LPBuffer<element::i4>& Constant::LPBuffer<element::i4>::operator++() {
    iter->operator++();
    return *this;
}

template <>
Constant::LPBuffer<element::nf4>& Constant::LPBuffer<element::nf4>::operator++() {
    iter->operator++();
    return *this;
}

#define CONSTANT_FILL_DATA(ET, SRC_TYPE)                                      \
    template <>                                                               \
    void Constant::fill_lp_data<element::Type_t::ET>(const SRC_TYPE& value) { \
        ov::op::fill_buffer<element::ET>(get_data_ptr_nc(), m_shape, value);  \
    }

CONSTANT_FILL_DATA(u1, bool)
CONSTANT_FILL_DATA(u1, char)
CONSTANT_FILL_DATA(u1, signed char)
CONSTANT_FILL_DATA(u1, unsigned char)
CONSTANT_FILL_DATA(u1, short)
CONSTANT_FILL_DATA(u1, unsigned short)
CONSTANT_FILL_DATA(u1, int)
CONSTANT_FILL_DATA(u1, unsigned int)
CONSTANT_FILL_DATA(u1, long)
CONSTANT_FILL_DATA(u1, unsigned long)
CONSTANT_FILL_DATA(u1, long long)
CONSTANT_FILL_DATA(u1, unsigned long long)
CONSTANT_FILL_DATA(u1, float8_e4m3)
CONSTANT_FILL_DATA(u1, float8_e5m2)
CONSTANT_FILL_DATA(u1, float16)
CONSTANT_FILL_DATA(u1, bfloat16)
CONSTANT_FILL_DATA(u1, float)
CONSTANT_FILL_DATA(u1, double)

CONSTANT_FILL_DATA(u4, bool)
CONSTANT_FILL_DATA(u4, char)
CONSTANT_FILL_DATA(u4, signed char)
CONSTANT_FILL_DATA(u4, unsigned char)
CONSTANT_FILL_DATA(u4, short)
CONSTANT_FILL_DATA(u4, unsigned short)
CONSTANT_FILL_DATA(u4, int)
CONSTANT_FILL_DATA(u4, unsigned int)
CONSTANT_FILL_DATA(u4, long)
CONSTANT_FILL_DATA(u4, unsigned long)
CONSTANT_FILL_DATA(u4, long long)
CONSTANT_FILL_DATA(u4, unsigned long long)
CONSTANT_FILL_DATA(u4, float8_e4m3)
CONSTANT_FILL_DATA(u4, float8_e5m2)
CONSTANT_FILL_DATA(u4, float16)
CONSTANT_FILL_DATA(u4, bfloat16)
CONSTANT_FILL_DATA(u4, float)
CONSTANT_FILL_DATA(u4, double)

CONSTANT_FILL_DATA(i4, bool)
CONSTANT_FILL_DATA(i4, char)
CONSTANT_FILL_DATA(i4, signed char)
CONSTANT_FILL_DATA(i4, unsigned char)
CONSTANT_FILL_DATA(i4, short)
CONSTANT_FILL_DATA(i4, unsigned short)
CONSTANT_FILL_DATA(i4, int)
CONSTANT_FILL_DATA(i4, unsigned int)
CONSTANT_FILL_DATA(i4, long)
CONSTANT_FILL_DATA(i4, unsigned long)
CONSTANT_FILL_DATA(i4, long long)
CONSTANT_FILL_DATA(i4, unsigned long long)
CONSTANT_FILL_DATA(i4, float8_e4m3)
CONSTANT_FILL_DATA(i4, float8_e5m2)
CONSTANT_FILL_DATA(i4, float16)
CONSTANT_FILL_DATA(i4, bfloat16)
CONSTANT_FILL_DATA(i4, float)
CONSTANT_FILL_DATA(i4, double)

CONSTANT_FILL_DATA(nf4, bool)
CONSTANT_FILL_DATA(nf4, char)
CONSTANT_FILL_DATA(nf4, signed char)
CONSTANT_FILL_DATA(nf4, unsigned char)
CONSTANT_FILL_DATA(nf4, short)
CONSTANT_FILL_DATA(nf4, unsigned short)
CONSTANT_FILL_DATA(nf4, int)
CONSTANT_FILL_DATA(nf4, unsigned int)
CONSTANT_FILL_DATA(nf4, long)
CONSTANT_FILL_DATA(nf4, unsigned long)
CONSTANT_FILL_DATA(nf4, long long)
CONSTANT_FILL_DATA(nf4, unsigned long long)
CONSTANT_FILL_DATA(nf4, float8_e4m3)
CONSTANT_FILL_DATA(nf4, float8_e5m2)
CONSTANT_FILL_DATA(nf4, float16)
CONSTANT_FILL_DATA(nf4, bfloat16)
CONSTANT_FILL_DATA(nf4, float)
CONSTANT_FILL_DATA(nf4, double)

#undef CONSTANT_FILL_DATA

#define CONSTANT_CAST_VECTOR(ET, DST_TYPE)                                                              \
    template <>                                                                                         \
    void Constant::cast_lp_vector<element::Type_t::ET, DST_TYPE>(std::vector<DST_TYPE> & output_vector, \
                                                                 size_t num_elements) const {           \
        ov::op::cast_buffer<element::ET>(get_data_ptr(), num_elements, output_vector);                  \
    }

CONSTANT_CAST_VECTOR(u1, bool)
CONSTANT_CAST_VECTOR(u1, char)
CONSTANT_CAST_VECTOR(u1, signed char)
CONSTANT_CAST_VECTOR(u1, unsigned char)
CONSTANT_CAST_VECTOR(u1, short)
CONSTANT_CAST_VECTOR(u1, unsigned short)
CONSTANT_CAST_VECTOR(u1, int)
CONSTANT_CAST_VECTOR(u1, unsigned int)
CONSTANT_CAST_VECTOR(u1, long)
CONSTANT_CAST_VECTOR(u1, unsigned long)
CONSTANT_CAST_VECTOR(u1, long long)
CONSTANT_CAST_VECTOR(u1, unsigned long long)
CONSTANT_CAST_VECTOR(u1, float16)
CONSTANT_CAST_VECTOR(u1, bfloat16)
CONSTANT_CAST_VECTOR(u1, float)
CONSTANT_CAST_VECTOR(u1, double)

CONSTANT_CAST_VECTOR(u4, bool)
CONSTANT_CAST_VECTOR(u4, char)
CONSTANT_CAST_VECTOR(u4, signed char)
CONSTANT_CAST_VECTOR(u4, unsigned char)
CONSTANT_CAST_VECTOR(u4, short)
CONSTANT_CAST_VECTOR(u4, unsigned short)
CONSTANT_CAST_VECTOR(u4, int)
CONSTANT_CAST_VECTOR(u4, unsigned int)
CONSTANT_CAST_VECTOR(u4, long)
CONSTANT_CAST_VECTOR(u4, unsigned long)
CONSTANT_CAST_VECTOR(u4, long long)
CONSTANT_CAST_VECTOR(u4, unsigned long long)
CONSTANT_CAST_VECTOR(u4, float16)
CONSTANT_CAST_VECTOR(u4, bfloat16)
CONSTANT_CAST_VECTOR(u4, float)
CONSTANT_CAST_VECTOR(u4, double)

CONSTANT_CAST_VECTOR(i4, bool)
CONSTANT_CAST_VECTOR(i4, char)
CONSTANT_CAST_VECTOR(i4, signed char)
CONSTANT_CAST_VECTOR(i4, unsigned char)
CONSTANT_CAST_VECTOR(i4, short)
CONSTANT_CAST_VECTOR(i4, unsigned short)
CONSTANT_CAST_VECTOR(i4, int)
CONSTANT_CAST_VECTOR(i4, unsigned int)
CONSTANT_CAST_VECTOR(i4, long)
CONSTANT_CAST_VECTOR(i4, unsigned long)
CONSTANT_CAST_VECTOR(i4, long long)
CONSTANT_CAST_VECTOR(i4, unsigned long long)
CONSTANT_CAST_VECTOR(i4, float16)
CONSTANT_CAST_VECTOR(i4, bfloat16)
CONSTANT_CAST_VECTOR(i4, float)
CONSTANT_CAST_VECTOR(i4, double)

#undef CONSTANT_CAST_VECTOR

#define CONSTANT_WRITE_BUFFER(ET, SRC_TYPE)                                                    \
    template <>                                                                                \
    void Constant::write_lp_buffer<element::Type_t::ET>(const std::vector<SRC_TYPE>& source) { \
        ov::op::write_buffer<element::ET>(source, get_data_ptr_nc());                          \
    }

CONSTANT_WRITE_BUFFER(u1, bool)
CONSTANT_WRITE_BUFFER(u1, char)
CONSTANT_WRITE_BUFFER(u1, signed char)
CONSTANT_WRITE_BUFFER(u1, unsigned char)
CONSTANT_WRITE_BUFFER(u1, short)
CONSTANT_WRITE_BUFFER(u1, unsigned short)
CONSTANT_WRITE_BUFFER(u1, int)
CONSTANT_WRITE_BUFFER(u1, unsigned int)
CONSTANT_WRITE_BUFFER(u1, long)
CONSTANT_WRITE_BUFFER(u1, unsigned long)
CONSTANT_WRITE_BUFFER(u1, long long)
CONSTANT_WRITE_BUFFER(u1, unsigned long long)
CONSTANT_WRITE_BUFFER(u1, float8_e4m3)
CONSTANT_WRITE_BUFFER(u1, float8_e5m2)
CONSTANT_WRITE_BUFFER(u1, float16)
CONSTANT_WRITE_BUFFER(u1, bfloat16)
CONSTANT_WRITE_BUFFER(u1, float)
CONSTANT_WRITE_BUFFER(u1, double)

CONSTANT_WRITE_BUFFER(u4, bool)
CONSTANT_WRITE_BUFFER(u4, char)
CONSTANT_WRITE_BUFFER(u4, signed char)
CONSTANT_WRITE_BUFFER(u4, unsigned char)
CONSTANT_WRITE_BUFFER(u4, short)
CONSTANT_WRITE_BUFFER(u4, unsigned short)
CONSTANT_WRITE_BUFFER(u4, int)
CONSTANT_WRITE_BUFFER(u4, unsigned int)
CONSTANT_WRITE_BUFFER(u4, long)
CONSTANT_WRITE_BUFFER(u4, unsigned long)
CONSTANT_WRITE_BUFFER(u4, long long)
CONSTANT_WRITE_BUFFER(u4, unsigned long long)
CONSTANT_WRITE_BUFFER(u4, float8_e4m3)
CONSTANT_WRITE_BUFFER(u4, float8_e5m2)
CONSTANT_WRITE_BUFFER(u4, float16)
CONSTANT_WRITE_BUFFER(u4, bfloat16)
CONSTANT_WRITE_BUFFER(u4, float)
CONSTANT_WRITE_BUFFER(u4, double)

CONSTANT_WRITE_BUFFER(i4, bool)
CONSTANT_WRITE_BUFFER(i4, char)
CONSTANT_WRITE_BUFFER(i4, signed char)
CONSTANT_WRITE_BUFFER(i4, unsigned char)
CONSTANT_WRITE_BUFFER(i4, short)
CONSTANT_WRITE_BUFFER(i4, unsigned short)
CONSTANT_WRITE_BUFFER(i4, int)
CONSTANT_WRITE_BUFFER(i4, unsigned int)
CONSTANT_WRITE_BUFFER(i4, long)
CONSTANT_WRITE_BUFFER(i4, unsigned long)
CONSTANT_WRITE_BUFFER(i4, long long)
CONSTANT_WRITE_BUFFER(i4, unsigned long long)
CONSTANT_WRITE_BUFFER(i4, float8_e4m3)
CONSTANT_WRITE_BUFFER(i4, float8_e5m2)
CONSTANT_WRITE_BUFFER(i4, float16)
CONSTANT_WRITE_BUFFER(i4, bfloat16)
CONSTANT_WRITE_BUFFER(i4, float)
CONSTANT_WRITE_BUFFER(i4, double)

CONSTANT_WRITE_BUFFER(nf4, bool)
CONSTANT_WRITE_BUFFER(nf4, char)
CONSTANT_WRITE_BUFFER(nf4, signed char)
CONSTANT_WRITE_BUFFER(nf4, unsigned char)
CONSTANT_WRITE_BUFFER(nf4, short)
CONSTANT_WRITE_BUFFER(nf4, unsigned short)
CONSTANT_WRITE_BUFFER(nf4, int)
CONSTANT_WRITE_BUFFER(nf4, unsigned int)
CONSTANT_WRITE_BUFFER(nf4, long)
CONSTANT_WRITE_BUFFER(nf4, unsigned long)
CONSTANT_WRITE_BUFFER(nf4, long long)
CONSTANT_WRITE_BUFFER(nf4, unsigned long long)
CONSTANT_WRITE_BUFFER(nf4, float8_e4m3)
CONSTANT_WRITE_BUFFER(nf4, float8_e5m2)
CONSTANT_WRITE_BUFFER(nf4, float16)
CONSTANT_WRITE_BUFFER(nf4, bfloat16)
CONSTANT_WRITE_BUFFER(nf4, float)
CONSTANT_WRITE_BUFFER(nf4, double)

#undef CONSTANT_WRITE_BUFFER

}  // namespace v0
}  // namespace op
}  // namespace ov
