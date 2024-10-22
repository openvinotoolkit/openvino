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
#include "openvino/core/tensor_util.hpp"
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

#define SUPPORTED_ET                                                                                                 \
    boolean, bf16, f16, f32, f64, i4, i8, i16, i32, i64, u1, u2, u3, u4, u6, u8, u16, u32, u64, nf4, f8e4m3, f8e5m2, \
        f4e2m1, f8e8m0

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

#ifdef __clang__
#    pragma clang diagnostic push
#    ifdef __has_warning
#        if __has_warning("-Wimplicit-const-int-float-conversion")
#            pragma clang diagnostic ignored "-Wimplicit-const-int-float-conversion"
#        elif __has_warning("-Wimplicit-int-float-conversion")
#            pragma clang diagnostic ignored "-Wimplicit-int-float-conversion"
#        endif
#    endif
#elif defined(__GNUC__)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wsign-compare"
#    pragma GCC diagnostic ignored "-Wbool-compare"
#elif defined(_MSC_VER)
#    pragma warning(push)
#    pragma warning(disable : 4018)
#    pragma warning(disable : 4804)
#endif
template <
    class U,
    class ConstantT,
    typename std::enable_if<!std::is_unsigned<ConstantT>::value && !std::is_same<U, ConstantT>::value>::type* = nullptr>
static bool in_type_range(const ConstantT v) {
    return std::numeric_limits<U>::lowest() <= v && v <= std::numeric_limits<U>::max();
}

template <
    class U,
    class ConstantT,
    typename std::enable_if<std::is_unsigned<ConstantT>::value && !std::is_same<U, ConstantT>::value>::type* = nullptr>
static bool in_type_range(const ConstantT v) {
    return v <= std::numeric_limits<U>::max();
}
#if defined(__clang__)
#    pragma clang diagnostic pop
#elif defined(__GNUC__)
#    pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#    pragma warning(pop)
#endif

template <element::Type_t ET, class U, typename std::enable_if<ET == element::u1>::type* = nullptr>
fundamental_type_for<ET> convert_if_in_element_range(const U& value) {
    using T = fundamental_type_for<ET>;
    return static_cast<T>(static_cast<bool>(value));
}

template <element::Type_t ET,
          class U,
          typename std::enable_if<ET == element::nf4 && !std::is_integral<U>::value>::type* = nullptr>
float convert_if_in_element_range(const U& value) {
    return static_cast<float>(value);
}

template <
    element::Type_t ET,
    class U,
    typename std::enable_if<ET == element::u4 || (ET == element::nf4 && std::is_integral<U>::value)>::type* = nullptr>
fundamental_type_for<ET> convert_if_in_element_range(const U& value) {
    using T = fundamental_type_for<ET>;
    auto result = static_cast<T>(value);
    OPENVINO_ASSERT(0 <= result && result <= 15, "assigned value out of range for u4");
    return result;
}

template <element::Type_t ET, class U, typename std::enable_if<ET == element::i4>::type* = nullptr>
fundamental_type_for<ET> convert_if_in_element_range(const U& value) {
    using T = fundamental_type_for<ET>;
    auto result = static_cast<T>(value);
    OPENVINO_ASSERT(-8 <= result && result <= 7, "assigned value out of range for i4");
    return result;
}

template <element::Type_t ET, class U, typename std::enable_if<ET == element::u2>::type* = nullptr>
fundamental_type_for<ET> convert_if_in_element_range(const U& value) {
    using T = fundamental_type_for<ET>;
    auto result = static_cast<T>(value);
    OPENVINO_ASSERT(0 <= result && result <= 3, "assigned value out of range for u2");
    return result;
}

template <element::Type_t ET, class U, typename std::enable_if<ET == element::u3>::type* = nullptr>
fundamental_type_for<ET> convert_if_in_element_range(const U& value) {
    using T = fundamental_type_for<ET>;
    auto result = static_cast<T>(value);
    OPENVINO_ASSERT(0 <= result && result <= 7, "assigned value out of range for u3");
    return result;
}

template <element::Type_t ET, class U, typename std::enable_if<ET == element::u6>::type* = nullptr>
fundamental_type_for<ET> convert_if_in_element_range(const U& value) {
    using T = fundamental_type_for<ET>;
    auto result = static_cast<T>(value);
    OPENVINO_ASSERT(0 <= result && result <= 63, "assigned value out of range for u6");
    return result;
}

template <element::Type_t ET, class U, typename std::enable_if<ET == element::f4e2m1>::type* = nullptr>
fundamental_type_for<ET> convert_if_in_element_range(const U& value) {
    using T = fundamental_type_for<ET>;
    return static_cast<T>(value);
}

template <element::Type_t ET, class T>
void fill_buffer(void* buffer, const Shape& shape, const T& value) {
    std::fill_n(element::iterator<ET>(buffer), shape_size(shape), convert_if_in_element_range<ET>(value));
}

template <element::Type_t ET, class T>
void write_buffer(const std::vector<T>& source, void* buffer) {
    std::transform(source.begin(), source.end(), element::iterator<ET>(buffer), convert_if_in_element_range<ET, T>);
}

Strides calc_byte_strides(const Shape& shape, const element::Type& et) {
    Strides strides;
    if (!shape.empty() && et.bitwidth() >= 8) {
        strides.resize(shape.size());
        strides.back() = et.size();
        std::transform(shape.crbegin(),
                       shape.crend() - 1,
                       strides.rbegin(),
                       strides.rbegin() + 1,
                       std::multiplies<size_t>());
    }
    return strides;
}
}  // namespace

namespace v0 {

Constant::Constant(const Tensor& tensor)
    : m_element_type{tensor.get_element_type()},
      m_shape{tensor.get_shape()},
      m_byte_strides{m_element_type.bitwidth() >= 8 ? tensor.get_strides() : Strides{}},
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
      m_shape(shape),
      m_byte_strides{calc_byte_strides(m_shape, m_element_type)} {
    allocate_buffer(memset_allocation);
    constructor_validate_and_infer_types();
}

void Constant::allocate_buffer(bool memset_allocation) {
    // memset_allocation flag is to switch on initialization of objects in memory for element::string type
    // and set memory to zero for numeric element types
    const auto num_elements = shape_size(m_shape);
    const auto byte_size = element::get_memory_size(m_element_type, num_elements);
    if (m_element_type == ov::element::string) {
        m_data = std::make_shared<StringAlignedBuffer>(num_elements, byte_size, host_alignment(), memset_allocation);
    } else {
        constexpr uint8_t init_value = 0;
        m_data = std::make_shared<AlignedBuffer>(byte_size, host_alignment());

        if (memset_allocation) {
            std::memset(m_data->get_ptr(), init_value, m_data->size());
        } else {
            set_unused_bits(m_data->get_ptr());
        }
    }
}

void Constant::set_unused_bits(void* buffer) const {
    const auto byte_size = m_data->size();

    if (byte_size > 0) {
        const auto num_elements = shape_size(m_shape);

        if (element::is_bit_type(m_element_type)) {
            constexpr size_t storage_unit_byte_size = 1;
            const auto not_aligned_elements = num_elements % (8 / m_element_type.bitwidth());
            const uint8_t not_used_bits_mask = 0xff >> (m_element_type.bitwidth() * not_aligned_elements);
            reinterpret_cast<uint8_t*>(buffer)[byte_size - storage_unit_byte_size] &= ~not_used_bits_mask;
        } else if (element::is_nibble_type(m_element_type) && (num_elements % 2)) {
            constexpr size_t storage_unit_byte_size = 1;
            reinterpret_cast<uint8_t*>(buffer)[byte_size - storage_unit_byte_size] &= 0x0FU;
        } else if (element::is_split_bit_type(m_element_type)) {
            constexpr size_t storage_unit_byte_size = 3;
            const auto num_values = (24U / m_element_type.bitwidth());
            const auto not_aligned_elements = num_elements % num_values;
            const uint16_t not_used_upper_mask = ~(0xffff >> (not_aligned_elements * (16U / num_values)));

            auto ptr = reinterpret_cast<uint8_t*>(buffer) + (byte_size - storage_unit_byte_size);
            ptr[0] &= not_used_upper_mask >> 8U;
            ptr[1] &= not_used_upper_mask & 0x00ff;
            ptr[2] &= ~(0xff >> (not_aligned_elements * (8U / num_values)));
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
        std::memcpy(get_data_ptr_nc(), data, element::get_memory_size(m_element_type, num_elements));
    }
}

Constant::Constant(const element::Type& type, const Shape& shape, const std::shared_ptr<ov::AlignedBuffer>& data)
    : m_element_type(type),
      m_shape(shape),
      m_byte_strides(calc_byte_strides(m_shape, m_element_type)),
      m_data(data) {
    constructor_validate_and_infer_types();
}

Constant::Constant(const Constant& other)
    : m_element_type{other.m_element_type},
      m_shape{other.m_shape},
      m_byte_strides{other.m_byte_strides},
      m_data{other.m_data},
      m_all_elements_bitwise_identical{other.m_all_elements_bitwise_identical.load()},
      m_all_elements_bitwise_identical_checked{other.m_all_elements_bitwise_identical_checked.load()},
      m_alloc_buffer_on_visit_attributes{other.m_alloc_buffer_on_visit_attributes} {
    constructor_validate_and_infer_types();
}

Constant::Constant(const Constant& other, const Shape& new_shape)
    : m_element_type{other.m_element_type},
      m_shape{new_shape},
      m_byte_strides{calc_byte_strides(m_shape, m_element_type)},
      m_data{other.m_data},
      m_all_elements_bitwise_identical{other.m_all_elements_bitwise_identical.load()},
      m_all_elements_bitwise_identical_checked{other.m_all_elements_bitwise_identical_checked.load()} {
    const auto new_size = shape_size(new_shape);
    const auto other_size = shape_size(other.m_shape);
    OPENVINO_ASSERT(other_size == new_size, "ov::Shape size ", new_size, " is not equal to ", other_size);
    constructor_validate_and_infer_types();
}

Constant::Constant(const element::Type& type, const Shape& shape, const void* data, std::shared_ptr<void> so)
    : Constant(
          type,
          shape,
          // Note: const_cast used to store pointer only
          std::make_shared<ov::SharedBuffer<std::shared_ptr<void>>>(reinterpret_cast<char*>(const_cast<void*>(data)),
                                                                    element::get_memory_size(type, shape_size(shape)),
                                                                    so)) {}

Constant::~Constant() = default;

struct ValueToString : ov::element::NotSupported<std::string> {
    using ov::element::NotSupported<std::string>::visit;

    template <ov::element::Type_t ET, typename std::enable_if<ET == element::f64>::type* = nullptr>
    static result_type visit(const void* const ptr, const size_t index) {
        return to_cpp_string(element::iterator<ET>(ptr)[index]);
    }

    template <ov::element::Type_t ET,
              typename std::enable_if<ov::is_floating_point<fundamental_type_for<ET>>() && element::is_byte_type(ET) &&
                                      ET != element::f64>::type* = nullptr>
    static result_type visit(const void* const ptr, const size_t index) {
        const auto it = element::iterator<ET>(ptr);
        return to_cpp_string<float>(it[index]);
    }

    template <ov::element::Type_t ET,
              typename std::enable_if<ov::is_floating_point<fundamental_type_for<ET>>() &&
                                      element::is_nibble_type(ET)>::type* = nullptr>
    static result_type visit(const void* const ptr, const size_t index) {
        const auto it = element::iterator<ET>(ptr);
        return to_cpp_string<float>(*(it + index));
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
                    u2,
                    u3,
                    u4,
                    u6,
                    u8,
                    u16,
                    u32,
                    u64,
                    nf4,
                    f8e4m3,
                    f8e5m2,
                    string,
                    f4e2m1,
                    f8e8m0>::apply<ValueToString>(get_element_type(), get_data_ptr(), index);
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
              typename std::enable_if<ET == element::f64>::type* = nullptr>
    static result_type visit(const void* const ptr, const size_t num_elements, std::vector<std::string>& strs) {
        const auto first = element::iterator<ET>(ptr);
        std::transform(first, first + num_elements, std::back_inserter(strs), to_cpp_string<double>);
    }

    template <ov::element::Type_t ET,
              class T = fundamental_type_for<ET>,
              typename std::enable_if<ov::is_floating_point<T>() && ET != element::f64>::type* = nullptr>
    static result_type visit(const void* const ptr, const size_t num_elements, std::vector<std::string>& strs) {
        const auto first = element::iterator<ET>(ptr);
        std::transform(first, first + num_elements, std::back_inserter(strs), to_cpp_string<float>);
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
    IfTypeOf<boolean,
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
             u2,
             u3,
             u4,
             u6,
             u8,
             u16,
             u32,
             u64,
             nf4,
             string,
             f4e2m1,
             f8e8m0>::apply<ValuesToString>(get_element_type(), get_data_ptr(), shape_size(m_shape), out);
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

bool Constant::can_constant_fold(const OutputVector& input_values) const {
    return false;
}

const Tensor Constant::get_tensor_view() const {
    return get_data_ptr() ? Tensor{m_element_type, m_shape, m_data->get_ptr(), m_byte_strides} : Tensor{};
}

const Strides& Constant::get_strides() const {
    OPENVINO_ASSERT(m_element_type.bitwidth() >= 8,
                    "Could not get strides for types with bit widths less then 8 bit. Type: ",
                    m_element_type);
    return m_byte_strides;
}

size_t Constant::get_num_elements_to_cast(const int64_t n) const {
    auto num_elements_in_shape = shape_size(m_shape);
    return (n < 0 ? num_elements_in_shape : std::min(static_cast<size_t>(n), num_elements_in_shape));
}

template <>
Constant::LPBuffer<element::u1>::LPBuffer(void* ptr)
    : iter{std::make_shared<lp_iter>(reinterpret_cast<ov::fundamental_type_for<element::u1>*>(ptr))} {}

template <>
Constant::LPBuffer<element::u2>::LPBuffer(void* ptr)
    : iter{std::make_shared<lp_iter>(reinterpret_cast<ov::fundamental_type_for<element::u2>*>(ptr))} {}

template <>
Constant::LPBuffer<element::u3>::LPBuffer(void* ptr)
    : iter{std::make_shared<lp_iter>(reinterpret_cast<ov::fundamental_type_for<element::u3>*>(ptr))} {}

template <>
Constant::LPBuffer<element::u4>::LPBuffer(void* ptr)
    : iter{std::make_shared<lp_iter>(reinterpret_cast<ov::fundamental_type_for<element::u4>*>(ptr))} {}

template <>
Constant::LPBuffer<element::u6>::LPBuffer(void* ptr)
    : iter{std::make_shared<lp_iter>(reinterpret_cast<ov::fundamental_type_for<element::u6>*>(ptr))} {}

template <>
Constant::LPBuffer<element::i4>::LPBuffer(void* ptr)
    : iter{std::make_shared<lp_iter>(reinterpret_cast<ov::fundamental_type_for<element::i4>*>(ptr))} {}

template <>
Constant::LPBuffer<element::nf4>::LPBuffer(void* ptr)
    : iter{std::make_shared<lp_iter>(reinterpret_cast<ov::fundamental_type_for<element::nf4>*>(ptr))} {}

template <>
Constant::LPBuffer<element::f4e2m1>::LPBuffer(void* ptr)
    : iter{std::make_shared<lp_iter>(reinterpret_cast<ov::fundamental_type_for<element::f4e2m1>*>(ptr))} {}

template <>
void Constant::LPBuffer<element::u1>::write(const float value) {
    iter->operator*() = convert_if_in_element_range<element::u1>(value);
}

template <>
void Constant::LPBuffer<element::u2>::write(const float value) {
    iter->operator*() = convert_if_in_element_range<element::u2>(value);
}

template <>
void Constant::LPBuffer<element::u3>::write(const float value) {
    iter->operator*() = convert_if_in_element_range<element::u3>(value);
}

template <>
void Constant::LPBuffer<element::u4>::write(const float value) {
    iter->operator*() = convert_if_in_element_range<element::u4>(value);
}

template <>
void Constant::LPBuffer<element::u6>::write(const float value) {
    iter->operator*() = convert_if_in_element_range<element::u6>(value);
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
void Constant::LPBuffer<element::f4e2m1>::write(const float value) {
    iter->operator*() = convert_if_in_element_range<element::f4e2m1>(value);
}

template <>
ov::fundamental_type_for<element::u1> Constant::LPBuffer<element::u1>::read() const {
    return iter->operator*();
}

template <>
ov::fundamental_type_for<element::u2> Constant::LPBuffer<element::u2>::read() const {
    return iter->operator*();
}

template <>
ov::fundamental_type_for<element::u3> Constant::LPBuffer<element::u3>::read() const {
    return iter->operator*();
}

template <>
ov::fundamental_type_for<element::u4> Constant::LPBuffer<element::u4>::read() const {
    return iter->operator*();
}

template <>
ov::fundamental_type_for<element::u6> Constant::LPBuffer<element::u6>::read() const {
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
ov::fundamental_type_for<element::f4e2m1> Constant::LPBuffer<element::f4e2m1>::read() const {
    return iter->operator*();
}

template <>
Constant::LPBuffer<element::u1>& Constant::LPBuffer<element::u1>::operator++() {
    iter->operator++();
    return *this;
}

template <>
Constant::LPBuffer<element::u2>& Constant::LPBuffer<element::u2>::operator++() {
    iter->operator++();
    return *this;
}

template <>
Constant::LPBuffer<element::u3>& Constant::LPBuffer<element::u3>::operator++() {
    iter->operator++();
    return *this;
}

template <>
Constant::LPBuffer<element::u4>& Constant::LPBuffer<element::u4>::operator++() {
    iter->operator++();
    return *this;
}

template <>
Constant::LPBuffer<element::u6>& Constant::LPBuffer<element::u6>::operator++() {
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

template <>
Constant::LPBuffer<element::f4e2m1>& Constant::LPBuffer<element::f4e2m1>::operator++() {
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
CONSTANT_FILL_DATA(u1, float8_e8m0)
CONSTANT_FILL_DATA(u1, float16)
CONSTANT_FILL_DATA(u1, bfloat16)
CONSTANT_FILL_DATA(u1, float)
CONSTANT_FILL_DATA(u1, double)

CONSTANT_FILL_DATA(u2, bool)
CONSTANT_FILL_DATA(u2, char)
CONSTANT_FILL_DATA(u2, signed char)
CONSTANT_FILL_DATA(u2, unsigned char)
CONSTANT_FILL_DATA(u2, short)
CONSTANT_FILL_DATA(u2, unsigned short)
CONSTANT_FILL_DATA(u2, int)
CONSTANT_FILL_DATA(u2, unsigned int)
CONSTANT_FILL_DATA(u2, long)
CONSTANT_FILL_DATA(u2, unsigned long)
CONSTANT_FILL_DATA(u2, long long)
CONSTANT_FILL_DATA(u2, unsigned long long)
CONSTANT_FILL_DATA(u2, float8_e4m3)
CONSTANT_FILL_DATA(u2, float8_e5m2)
CONSTANT_FILL_DATA(u2, float8_e8m0)
CONSTANT_FILL_DATA(u2, float16)
CONSTANT_FILL_DATA(u2, bfloat16)
CONSTANT_FILL_DATA(u2, float)
CONSTANT_FILL_DATA(u2, double)

CONSTANT_FILL_DATA(u3, bool)
CONSTANT_FILL_DATA(u3, char)
CONSTANT_FILL_DATA(u3, signed char)
CONSTANT_FILL_DATA(u3, unsigned char)
CONSTANT_FILL_DATA(u3, short)
CONSTANT_FILL_DATA(u3, unsigned short)
CONSTANT_FILL_DATA(u3, int)
CONSTANT_FILL_DATA(u3, unsigned int)
CONSTANT_FILL_DATA(u3, long)
CONSTANT_FILL_DATA(u3, unsigned long)
CONSTANT_FILL_DATA(u3, long long)
CONSTANT_FILL_DATA(u3, unsigned long long)
CONSTANT_FILL_DATA(u3, float8_e4m3)
CONSTANT_FILL_DATA(u3, float8_e5m2)
CONSTANT_FILL_DATA(u3, float8_e8m0)
CONSTANT_FILL_DATA(u3, float16)
CONSTANT_FILL_DATA(u3, bfloat16)
CONSTANT_FILL_DATA(u3, float)
CONSTANT_FILL_DATA(u3, double)

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
CONSTANT_FILL_DATA(u4, float8_e8m0)
CONSTANT_FILL_DATA(u4, float16)
CONSTANT_FILL_DATA(u4, bfloat16)
CONSTANT_FILL_DATA(u4, float)
CONSTANT_FILL_DATA(u4, double)

CONSTANT_FILL_DATA(u6, bool)
CONSTANT_FILL_DATA(u6, char)
CONSTANT_FILL_DATA(u6, signed char)
CONSTANT_FILL_DATA(u6, unsigned char)
CONSTANT_FILL_DATA(u6, short)
CONSTANT_FILL_DATA(u6, unsigned short)
CONSTANT_FILL_DATA(u6, int)
CONSTANT_FILL_DATA(u6, unsigned int)
CONSTANT_FILL_DATA(u6, long)
CONSTANT_FILL_DATA(u6, unsigned long)
CONSTANT_FILL_DATA(u6, long long)
CONSTANT_FILL_DATA(u6, unsigned long long)
CONSTANT_FILL_DATA(u6, float8_e4m3)
CONSTANT_FILL_DATA(u6, float8_e5m2)
CONSTANT_FILL_DATA(u6, float8_e8m0)
CONSTANT_FILL_DATA(u6, float16)
CONSTANT_FILL_DATA(u6, bfloat16)
CONSTANT_FILL_DATA(u6, float)
CONSTANT_FILL_DATA(u6, double)

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
CONSTANT_FILL_DATA(i4, float8_e8m0)
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
CONSTANT_FILL_DATA(nf4, float8_e8m0)
CONSTANT_FILL_DATA(nf4, float16)
CONSTANT_FILL_DATA(nf4, bfloat16)
CONSTANT_FILL_DATA(nf4, float)
CONSTANT_FILL_DATA(nf4, double)

CONSTANT_FILL_DATA(f4e2m1, bool)
CONSTANT_FILL_DATA(f4e2m1, char)
CONSTANT_FILL_DATA(f4e2m1, signed char)
CONSTANT_FILL_DATA(f4e2m1, unsigned char)
CONSTANT_FILL_DATA(f4e2m1, short)
CONSTANT_FILL_DATA(f4e2m1, unsigned short)
CONSTANT_FILL_DATA(f4e2m1, int)
CONSTANT_FILL_DATA(f4e2m1, unsigned int)
CONSTANT_FILL_DATA(f4e2m1, long)
CONSTANT_FILL_DATA(f4e2m1, unsigned long)
CONSTANT_FILL_DATA(f4e2m1, long long)
CONSTANT_FILL_DATA(f4e2m1, unsigned long long)
CONSTANT_FILL_DATA(f4e2m1, float8_e4m3)
CONSTANT_FILL_DATA(f4e2m1, float8_e5m2)
CONSTANT_FILL_DATA(f4e2m1, float8_e8m0)
CONSTANT_FILL_DATA(f4e2m1, float16)
CONSTANT_FILL_DATA(f4e2m1, bfloat16)
CONSTANT_FILL_DATA(f4e2m1, float)
CONSTANT_FILL_DATA(f4e2m1, double)

#undef CONSTANT_FILL_DATA

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
CONSTANT_WRITE_BUFFER(u1, float8_e8m0)
CONSTANT_WRITE_BUFFER(u1, float16)
CONSTANT_WRITE_BUFFER(u1, bfloat16)
CONSTANT_WRITE_BUFFER(u1, float)
CONSTANT_WRITE_BUFFER(u1, double)

CONSTANT_WRITE_BUFFER(u2, bool)
CONSTANT_WRITE_BUFFER(u2, char)
CONSTANT_WRITE_BUFFER(u2, signed char)
CONSTANT_WRITE_BUFFER(u2, unsigned char)
CONSTANT_WRITE_BUFFER(u2, short)
CONSTANT_WRITE_BUFFER(u2, unsigned short)
CONSTANT_WRITE_BUFFER(u2, int)
CONSTANT_WRITE_BUFFER(u2, unsigned int)
CONSTANT_WRITE_BUFFER(u2, long)
CONSTANT_WRITE_BUFFER(u2, unsigned long)
CONSTANT_WRITE_BUFFER(u2, long long)
CONSTANT_WRITE_BUFFER(u2, unsigned long long)
CONSTANT_WRITE_BUFFER(u2, float8_e4m3)
CONSTANT_WRITE_BUFFER(u2, float8_e5m2)
CONSTANT_WRITE_BUFFER(u2, float8_e8m0)
CONSTANT_WRITE_BUFFER(u2, float16)
CONSTANT_WRITE_BUFFER(u2, bfloat16)
CONSTANT_WRITE_BUFFER(u2, float)
CONSTANT_WRITE_BUFFER(u2, double)

CONSTANT_WRITE_BUFFER(u3, bool)
CONSTANT_WRITE_BUFFER(u3, char)
CONSTANT_WRITE_BUFFER(u3, signed char)
CONSTANT_WRITE_BUFFER(u3, unsigned char)
CONSTANT_WRITE_BUFFER(u3, short)
CONSTANT_WRITE_BUFFER(u3, unsigned short)
CONSTANT_WRITE_BUFFER(u3, int)
CONSTANT_WRITE_BUFFER(u3, unsigned int)
CONSTANT_WRITE_BUFFER(u3, long)
CONSTANT_WRITE_BUFFER(u3, unsigned long)
CONSTANT_WRITE_BUFFER(u3, long long)
CONSTANT_WRITE_BUFFER(u3, unsigned long long)
CONSTANT_WRITE_BUFFER(u3, float8_e4m3)
CONSTANT_WRITE_BUFFER(u3, float8_e5m2)
CONSTANT_WRITE_BUFFER(u3, float8_e8m0)
CONSTANT_WRITE_BUFFER(u3, float16)
CONSTANT_WRITE_BUFFER(u3, bfloat16)
CONSTANT_WRITE_BUFFER(u3, float)
CONSTANT_WRITE_BUFFER(u3, double)

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
CONSTANT_WRITE_BUFFER(u4, float8_e8m0)
CONSTANT_WRITE_BUFFER(u4, float16)
CONSTANT_WRITE_BUFFER(u4, bfloat16)
CONSTANT_WRITE_BUFFER(u4, float)
CONSTANT_WRITE_BUFFER(u4, double)

CONSTANT_WRITE_BUFFER(u6, bool)
CONSTANT_WRITE_BUFFER(u6, char)
CONSTANT_WRITE_BUFFER(u6, signed char)
CONSTANT_WRITE_BUFFER(u6, unsigned char)
CONSTANT_WRITE_BUFFER(u6, short)
CONSTANT_WRITE_BUFFER(u6, unsigned short)
CONSTANT_WRITE_BUFFER(u6, int)
CONSTANT_WRITE_BUFFER(u6, unsigned int)
CONSTANT_WRITE_BUFFER(u6, long)
CONSTANT_WRITE_BUFFER(u6, unsigned long)
CONSTANT_WRITE_BUFFER(u6, long long)
CONSTANT_WRITE_BUFFER(u6, unsigned long long)
CONSTANT_WRITE_BUFFER(u6, float8_e4m3)
CONSTANT_WRITE_BUFFER(u6, float8_e5m2)
CONSTANT_WRITE_BUFFER(u6, float8_e8m0)
CONSTANT_WRITE_BUFFER(u6, float16)
CONSTANT_WRITE_BUFFER(u6, bfloat16)
CONSTANT_WRITE_BUFFER(u6, float)
CONSTANT_WRITE_BUFFER(u6, double)

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
CONSTANT_WRITE_BUFFER(i4, float8_e8m0)
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
CONSTANT_WRITE_BUFFER(nf4, float8_e8m0)
CONSTANT_WRITE_BUFFER(nf4, float16)
CONSTANT_WRITE_BUFFER(nf4, bfloat16)
CONSTANT_WRITE_BUFFER(nf4, float)
CONSTANT_WRITE_BUFFER(nf4, double)

CONSTANT_WRITE_BUFFER(f4e2m1, bool)
CONSTANT_WRITE_BUFFER(f4e2m1, char)
CONSTANT_WRITE_BUFFER(f4e2m1, signed char)
CONSTANT_WRITE_BUFFER(f4e2m1, unsigned char)
CONSTANT_WRITE_BUFFER(f4e2m1, short)
CONSTANT_WRITE_BUFFER(f4e2m1, unsigned short)
CONSTANT_WRITE_BUFFER(f4e2m1, int)
CONSTANT_WRITE_BUFFER(f4e2m1, unsigned int)
CONSTANT_WRITE_BUFFER(f4e2m1, long)
CONSTANT_WRITE_BUFFER(f4e2m1, unsigned long)
CONSTANT_WRITE_BUFFER(f4e2m1, long long)
CONSTANT_WRITE_BUFFER(f4e2m1, unsigned long long)
CONSTANT_WRITE_BUFFER(f4e2m1, float8_e4m3)
CONSTANT_WRITE_BUFFER(f4e2m1, float8_e5m2)
CONSTANT_WRITE_BUFFER(f4e2m1, float8_e8m0)
CONSTANT_WRITE_BUFFER(f4e2m1, float16)
CONSTANT_WRITE_BUFFER(f4e2m1, bfloat16)
CONSTANT_WRITE_BUFFER(f4e2m1, float)
CONSTANT_WRITE_BUFFER(f4e2m1, double)

#undef CONSTANT_WRITE_BUFFER

template <class U>
struct ElementConvert : element::NotSupported<void> {
    using element::NotSupported<void>::visit;

    template <element::Type_t ET,
              class InputIt,
              class OutputIt,
              typename std::enable_if<ET != element::string>::type* = nullptr>
    static result_type visit(const InputIt src, OutputIt dst, const size_t n) {
        auto first = element::iterator<ET>(src);
        reference::convert(first, dst, n);
    }

    template <element::Type_t ET,
              class InputIt,
              class OutputIt,
              typename std::enable_if<ET == element::string>::type* = nullptr>
    [[noreturn]] static result_type visit(const InputIt, OutputIt, const size_t) {
        OPENVINO_THROW("'cast_vector' does not support casting Constant of type ",
                       ET,
                       " into std::vector of ",
                       element::from<U>());
    }
};

template <>
struct ElementConvert<bool> : element::NotSupported<void> {
    using element::NotSupported<void>::visit;

    template <element::Type_t ET,
              class InputIt,
              class OutputIt,
              typename std::enable_if<ET != element::string>::type* = nullptr>
    static result_type visit(InputIt src, OutputIt dst, const size_t n) {
        auto first = element::iterator<ET>(src);
        using T = ov::fundamental_type_for<ET>;
        std::transform(first, first + n, dst, [](const T v) {
            return static_cast<bool>(v);
        });
    }

    template <element::Type_t ET,
              class InputIt,
              class OutputIt,
              typename std::enable_if<ET == element::string>::type* = nullptr>
    [[noreturn]] static result_type visit(InputIt, OutputIt, const size_t) {
        OPENVINO_THROW("'cast_vector' does not support casting Constant of type ", ET, " into std::vector of boolean");
    }
};

#define CONSTANT_CAST_VECTOR(DTYPE)                                                     \
    template <>                                                                         \
    OPENVINO_API std::vector<DTYPE> Constant::cast_vector(int64_t num_elements) const { \
        std::vector<DTYPE> output(get_num_elements_to_cast(num_elements));              \
        using namespace ov::element;                                                    \
        IfTypeOf<SUPPORTED_ET>::apply<ElementConvert<DTYPE>>(m_element_type,            \
                                                             get_data_ptr(),            \
                                                             output.data(),             \
                                                             output.size());            \
        return output;                                                                  \
    }

template <>
OPENVINO_API std::vector<bool> Constant::cast_vector(int64_t num_elements) const {
    std::vector<bool> output(get_num_elements_to_cast(num_elements));
    using namespace ov::element;
    IfTypeOf<SUPPORTED_ET>::apply<ElementConvert<bool>>(m_element_type, get_data_ptr(), output.begin(), output.size());
    return output;
}

CONSTANT_CAST_VECTOR(char)
CONSTANT_CAST_VECTOR(signed char)
CONSTANT_CAST_VECTOR(unsigned char)
CONSTANT_CAST_VECTOR(short)
CONSTANT_CAST_VECTOR(unsigned short)
CONSTANT_CAST_VECTOR(int)
CONSTANT_CAST_VECTOR(unsigned int)
CONSTANT_CAST_VECTOR(long)
CONSTANT_CAST_VECTOR(unsigned long)
CONSTANT_CAST_VECTOR(long long)
CONSTANT_CAST_VECTOR(unsigned long long)
CONSTANT_CAST_VECTOR(float16)
CONSTANT_CAST_VECTOR(bfloat16)
CONSTANT_CAST_VECTOR(float)
CONSTANT_CAST_VECTOR(double)

}  // namespace v0
}  // namespace op
}  // namespace ov
