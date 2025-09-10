// Copyright (C) 2018-2025 Intel Corporation
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
#include "openvino/core/memory_util.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/core/tensor_util.hpp"
#include "openvino/core/type/element_iterator.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/core/type/nf4.hpp"
#include "openvino/reference/convert.hpp"
#include "openvino/reference/utils/type_util.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "openvino/runtime/string_aligned_buffer.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/util/variant_visitor.hpp"

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
#    ifdef __has_warningop
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
template <element::Type_t ET, class U>
bool in_t_range(const U& v) {
    // return true;
    using ConstantT = fundamental_type_for<ET>;
    if constexpr (ET == element::u1) {
        return true;
    } else if constexpr (ET == element::u2) {
        const auto temp = static_cast<ConstantT>(v);
        return 0 <= temp && temp <= 3;
    } else if constexpr (ET == element::u3) {
        const auto temp = static_cast<ConstantT>(v);
        return 0 <= temp && temp <= 7;
    } else if constexpr (ET == element::u4 || (ET == element::nf4 && std::is_integral_v<U>)) {
        const auto temp = static_cast<ConstantT>(v);
        return 0 <= temp && temp <= 15;
    } else if constexpr (ET == element::u6) {
        const auto temp = static_cast<ConstantT>(v);
        return 0 <= temp && temp <= 63;
    } else if constexpr (ET == element::i4) {
        const auto temp = static_cast<ConstantT>(v);
        return -8 <= temp && temp <= 7;
    } else if (ET == element::nf4 && !std::is_integral_v<U>) {
        return true;
    } else if (ET == element::f4e2m1) {
        return true;
    } else if constexpr (std::is_same_v<U, ConstantT>) {
        return true;
    } else if constexpr (std::is_unsigned_v<ConstantT> && std::is_integral_v<U>) {
        if constexpr (std::numeric_limits<ConstantT>::max() < std::numeric_limits<U>::max()) {
            return true;
        } else {
            return v <= std::numeric_limits<U>::max();
        }
    } else if constexpr (std::is_unsigned_v<ConstantT>) {
        return v <= std::numeric_limits<U>::max();
    } else if constexpr (std::is_integral_v<ConstantT> && std::is_integral_v<U>) {
        if constexpr (std::numeric_limits<U>::lowest() < std::numeric_limits<ConstantT>::lowest() &&
                      std::numeric_limits<U>::max() > std::numeric_limits<ConstantT>::max()) {
            return true;
        } else if constexpr (std::numeric_limits<ConstantT>::lowest() < std::numeric_limits<U>::lowest() &&
                             std::numeric_limits<U>::max() <= std::numeric_limits<ConstantT>::max()) {
            return std::numeric_limits<U>::lowest() <= v;
        } else if constexpr (std::numeric_limits<ConstantT>::lowest() >= std::numeric_limits<U>::lowest() &&
                             std::numeric_limits<U>::max() > std::numeric_limits<ConstantT>::max()) {
            return v <= std::numeric_limits<U>::max();
        } else {
            return std::numeric_limits<U>::lowest() <= v && v <= std::numeric_limits<U>::max();
        }
    } else {
        return std::numeric_limits<U>::lowest() <= v && v <= std::numeric_limits<U>::max();
    }
}

template <element::Type_t ET, class U, bool enable_validation = true>
auto convert_if_in_range(const U& value) {
    if constexpr (enable_validation) {
        OPENVINO_ASSERT(in_t_range<ET>(value), value, " assigned value out of range for ", ET);
    }
    if constexpr (ET == element::u1) {
        return static_cast<fundamental_type_for<ET>>(static_cast<bool>(value));
    } else if constexpr (ET == element::nf4 && !std::is_integral_v<U>) {
        return static_cast<float>(value);
    } else {
        return static_cast<fundamental_type_for<ET>>(value);
    }
}

#if defined(__clang__)
#    pragma clang diagnostic pop
#elif defined(__GNUC__)
#    pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#    pragma warning(pop)
#endif

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
      // cast is for internal use only to store tensor data in shared buffer (not for modification)
      m_data{std::make_shared<SharedBuffer<Tensor>>(const_cast<char*>(static_cast<const char*>(tensor.data())),
                                                    tensor.get_byte_size(),
                                                    tensor)} {
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
    const auto byte_size = ov::util::get_memory_size_safe(m_element_type, m_shape);
    OPENVINO_ASSERT(byte_size, "Cannot allocate memory for type: ", m_element_type, " and shape: ", m_shape);
    if (m_element_type == ov::element::string) {
        const auto num_elements = shape_size(m_shape);
        m_data = std::make_shared<StringAlignedBuffer>(num_elements, *byte_size, host_alignment(), memset_allocation);
    } else {
        constexpr uint8_t init_value = 0;
        m_data = std::make_shared<AlignedBuffer>(*byte_size, host_alignment());

        // AlignedBuffer allocates 1 byte for empty constants, and we set it to zero
        if (memset_allocation || *byte_size == 0) {
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
            const uint8_t not_used_bits_mask = element::is_lsb_packed(m_element_type)
                                                   ? 0xff << (m_element_type.bitwidth() * not_aligned_elements)
                                                   : 0xff >> (m_element_type.bitwidth() * not_aligned_elements);
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
        std::memcpy(get_data_ptr_nc(), data, ov::util::get_memory_size(m_element_type, num_elements));
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
                                                                    ov::util::get_memory_size(type, shape_size(shape)),
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
    return IfTypeOf<SUPPORTED_ET, string>::apply<ValueToString>(get_element_type(), get_data_ptr(), index);
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
    IfTypeOf<SUPPORTED_ET, string>::apply<ValuesToString>(get_element_type(), get_data_ptr(), shape_size(m_shape), out);
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
    if (!outputs.empty() && outputs[0].get_element_type() != m_element_type)
        return evaluate(outputs, {});  // for TypeRelaxed<Constant>
    outputs.resize(1);
    outputs[0] = get_tensor_view();
    return get_data_ptr() != nullptr;
}

bool Constant::evaluate_upper(TensorVector& outputs) const {
    if (!outputs.empty() && outputs[0].get_element_type() != m_element_type)
        return evaluate(outputs, {});  // for TypeRelaxed<Constant>
    outputs.resize(1);
    outputs[0] = get_tensor_view();
    return get_data_ptr() != nullptr;
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

struct ConstCast : element::NotSupported<void> {
    using element::NotSupported<void>::visit;

    template <element::Type_t ET, class OutputIt>
    static result_type visit(const void* src, const size_t n, OutputIt dst) {
        reference::convert(element::iterator<ET>(src), dst, n);
    }
};

struct ConstFill : element::NotSupported<void> {
    using element::NotSupported<void>::visit;

    template <element::Type_t ET, class T>
    static result_type visit(void* dst, const size_t n, const T& value) {
        std::fill_n(element::iterator<ET>(dst), n, convert_if_in_range<ET, T>(value));
    }
};

struct ConstWrite : element::NotSupported<void> {
    using element::NotSupported<void>::visit;

    template <element::Type_t ET, class InputIt>
    static result_type visit(const InputIt src, const size_t n, void* dst) {
        using T = std::decay_t<typename std::iterator_traits<InputIt>::value_type>;
        constexpr auto enable_validation = !element::is_byte_type(ET);
        std::transform(src, src + n, element::iterator<ET>(dst), convert_if_in_range<ET, T, enable_validation>);
    }
};
template <class T, class = void>
struct has_indirection : std::false_type {};

template <class T>
struct has_indirection<T, std::void_t<decltype(*std::declval<T>())>> : std::true_type {};

template <class T>
constexpr bool has_indirection_v = has_indirection<T>::value;

const auto element_type_from_variant = ov::util::VariantVisitor{[](auto&& value) -> element::Type {
    if constexpr (has_indirection_v<decltype(value)>) {
        return element::from<std::decay_t<decltype(*value)>>();
    } else {
        return element::from<decltype(value)>();
    }
}};

template <>
OPENVINO_API void Constant::data::fill(const element::Type& type, void* dst, const size_t n, const value& value) {
    if (const auto is_str = (type == element::string); is_str == std::holds_alternative<std::string>(value)) {
        const auto fill_visitor =
            ov::util::VariantVisitor{[&type, dst, n](const auto& v) {
                                         using namespace ov::element;
                                         IfTypeOf<SUPPORTED_ET>::apply<ConstFill>(type, dst, n, v);
                                     },
                                     [dst, n](const std::string& v) {
                                         std::uninitialized_fill_n(element::iterator<element::string>(dst), n, v);
                                     }};
        std::visit(fill_visitor, value);
    } else {
        if (is_str) {
            std::uninitialized_fill_n(element::iterator<element::string>(dst), n, std::string{});
        }
        OPENVINO_THROW("Constant does not support writing elements of type '",
                       std::visit(element_type_from_variant, value),
                       "' into Constant of type '",
                       type,
                       "'");
    }
}

template <>
OPENVINO_API void Constant::data::copy_n(const element::Type& type,
                                         const const_pointer src,
                                         const size_t n,
                                         void* dst) {
    if (const auto is_str = (type == element::string); is_str == std::holds_alternative<const std::string*>(src)) {
        const auto write_visitor =
            ov::util::VariantVisitor{[&type, n, dst](const auto src) {
                                         using namespace ov::element;
                                         IfTypeOf<SUPPORTED_ET>::apply<ConstWrite>(type, src, n, dst);
                                     },
                                     [n, dst](const std::string* src) {
                                         std::uninitialized_copy_n(src, n, element::iterator<element::string>(dst));
                                     }};
        std::visit(write_visitor, src);
    } else {
        if (is_str) {
            std::uninitialized_fill_n(element::iterator<element::string>(dst), n, std::string{});
        }
        OPENVINO_THROW("Constant does not support writing elements of type '",
                       std::visit(element_type_from_variant, src),
                       "' into Constant of type '",
                       type,
                       "'");
    }
}

template <>
OPENVINO_API void Constant::data::cast_n(const element::Type& type, const void* src, const size_t n, pointer dst) {
    const auto is_vistable = (type == element::string) == std::holds_alternative<std::string*>(dst);
    OPENVINO_ASSERT(is_vistable,
                    "Constant does not support casting elements of type '",
                    type,
                    "' into std::vector of type '",
                    std::visit(element_type_from_variant, dst),
                    "'");
    const auto cast_visitor =
        ov::util::VariantVisitor{[&type, src, n](auto dst) {
                                     using namespace ov::element;
                                     IfTypeOf<SUPPORTED_ET>::apply<ConstCast>(type, src, n, dst);
                                 },
                                 [src, n](std::string* dst) {
                                     std::uninitialized_copy_n(element::iterator<element::string>(src), n, dst);
                                 }};
    std::visit(cast_visitor, dst);
}
}  // namespace v0
}  // namespace op
}  // namespace ov
