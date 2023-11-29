// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstring>

#ifndef IN_OV_COMPONENT
#    define IN_OV_COMPONENT
#    define WAS_OV_LIBRARY_DEFINED_CONSTANT
#endif

#include "ngraph/runtime/shared_buffer.hpp"

#ifdef WAS_OV_LIBRARY_DEFINED_CONSTANT
#    undef IN_OV_COMPONENT
#    undef WAS_OV_LIBRARY_DEFINED_CONSTANT
#endif
#include "openvino/core/coordinate_diff.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/element_type_traits.hpp"

namespace ov {

class AlignedBuffer;

namespace op {
namespace v0 {
/// \brief Class for constants.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Constant : public Op {
public:
    OPENVINO_OP("Constant", "opset1");

    Constant() = default;

    OPENVINO_SUPPRESS_DEPRECATED_START
    /// \brief Initialize a constant from tensor
    /// \param tensor The tensor with data
    OPENVINO_DEPRECATED("This constructor is deprecated and will be removed in 2024.0 release")
    Constant(const std::shared_ptr<ngraph::runtime::Tensor>& tensor);

    /// \brief Constructs a tensor constant with the supplied data
    ///
    /// \param type The element type of the tensor constant.
    /// \param shape The shape of the tensor constant.
    /// \param data A pointer to pre-allocated shared data.
    template <typename T>
    OPENVINO_DEPRECATED("This constructor is deprecated and will be removed in 2024.0 release")
    Constant(const element::Type& type, const Shape& shape, std::shared_ptr<ngraph::runtime::SharedBuffer<T>> data)
        : m_element_type(type),
          m_shape(shape) {
        m_data = legacy_to_ov_aligned_buffer(data);
        constructor_validate_and_infer_types();
    }
    OPENVINO_SUPPRESS_DEPRECATED_END

    /// \brief Initialize a constant from ov::Tensor
    /// \param tensor The ov::Tensor with data
    Constant(const ov::Tensor& tensor);

    /// \brief Constructs a tensor constant.
    ///
    /// \param type The element type of the tensor constant.
    /// \param shape The shape of the tensor constant.
    /// \param values A vector of literals for initializing the tensor constant. The
    ///               size of values must match the size of the shape.
    template <typename T>
    Constant(const element::Type& type, const Shape& shape, const std::vector<T>& values)
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

        if (values.size() == 1) {
            fill_data(type, values.front());
        } else {
            write_values(values);
        }
    }

    /// \brief Create uninitialized constant
    Constant(const element::Type& type, const Shape& shape);
    /// \brief Constructs a uniform tensor constant.
    ///
    /// \param type The element type of the tensor constant.
    /// \param shape The shape of the tensor constant.
    /// \param value A scalar for initializing the uniform tensor constant. The
    ///               value is broadcast to the specified shape.
    template <class T, class = typename std::enable_if<std::is_fundamental<T>::value>::type>
    Constant(const element::Type& type, const Shape& shape, T value) : Constant(false, type, shape) {
        fill_data(type, value);
    }

    template <typename T>
    void fill_data(const element::Type& type, T value) {
        using Type_t = element::Type_t;
#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic error "-Wswitch"
#    pragma GCC diagnostic error "-Wswitch-enum"
#endif
        switch (type) {
        case Type_t::boolean:
            fill_data<Type_t::boolean>(value);
            break;
        case Type_t::bf16:
            fill_data<Type_t::bf16>(value);
            break;
        case Type_t::f16:
            fill_data<Type_t::f16>(value);
            break;
        case Type_t::f32:
            fill_data<Type_t::f32>(value);
            break;
        case Type_t::f64:
            fill_data<Type_t::f64>(value);
            break;
        case Type_t::i4:
            fill_data<Type_t::i4>(value);
            break;
        case Type_t::i8:
            fill_data<Type_t::i8>(value);
            break;
        case Type_t::i16:
            fill_data<Type_t::i16>(value);
            break;
        case Type_t::i32:
            fill_data<Type_t::i32>(value);
            break;
        case Type_t::i64:
            fill_data<Type_t::i64>(value);
            break;
        case Type_t::u1:
            fill_data<Type_t::u1>(value);
            break;
        case Type_t::u4:
            fill_data<Type_t::u4>(value);
            break;
        case Type_t::u8:
            fill_data<Type_t::u8>(value);
            break;
        case Type_t::u16:
            fill_data<Type_t::u16>(value);
            break;
        case Type_t::u32:
            fill_data<Type_t::u32>(value);
            break;
        case Type_t::u64:
            fill_data<Type_t::u64>(value);
            break;
        case Type_t::nf4:
            fill_data<Type_t::nf4>(value);
            break;
        case Type_t::string:
            fill_data<Type_t::string>(value);
            break;
        case Type_t::undefined:
        case Type_t::dynamic:
            OPENVINO_THROW("unsupported type");
        }
#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#    pragma GCC diagnostic pop
#endif
    }

    /// \brief Constructs a tensor constant
    ///        This constructor is mainly to support deserialization of constants.
    ///
    /// \param type The element type of the tensor constant.
    /// \param shape The shape of the tensor constant.
    /// \param values A list of string values to use as the constant data.
    Constant(const element::Type& type, const Shape& shape, const std::vector<std::string>& values);

    /// \brief Constructs a tensor constant with the supplied data
    ///
    /// \param type The element type of the tensor constant.
    /// \param shape The shape of the tensor constant.
    /// \param data A void* to constant data.
    Constant(const element::Type& type, const Shape& shape, const void* data);

    Constant(const element::Type& type, const Shape& shape, const std::shared_ptr<ov::AlignedBuffer>& data)
        : m_element_type(type),
          m_shape(shape) {
        m_data = data;
        constructor_validate_and_infer_types();
    }

    Constant(const Constant& other);
    Constant(const Constant& other, const Shape& new_shape);
    Constant& operator=(const Constant&) = delete;

    ~Constant() override;

    void validate_and_infer_types() override {
        infer_element_type();
        set_output_type(0, m_element_type, m_shape);
    }

    bool visit_attributes(AttributeVisitor& visitor) override;

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;
    bool has_evaluate() const override;
    bool evaluate_lower(TensorVector& outputs) const override;
    bool evaluate_upper(TensorVector& outputs) const override;

    // Don't constant fold a constant; it would make a copy
    bool constant_fold(OutputVector& outputs, const OutputVector& inputs) override {
        (void)outputs;
        (void)inputs;
        return false;
    }

    /// \brief Returns the value of the constant node as a Shape object
    ///        Can only be used on element::i64 nodes and interprets
    ///        negative values as zeros.
    Shape get_shape_val() const;
    /// \brief Returns the value of the constant node as a Strides
    ///        object
    ///        Can only be used on element::i64 nodes and interprets
    ///        negative values as zeros.
    Strides get_strides_val() const;
    /// \brief Returns the value of the constant node as a Coordinate
    ///        object
    ///        Can only be used on element::i64 nodes and interprets
    ///        negative values as zeros.
    Coordinate get_coordinate_val() const;
    /// \brief Returns the value of the constant node as a
    ///        CoordinateDiff object
    ///        Can only be used on element::i64 nodes.
    CoordinateDiff get_coordinate_diff_val() const;
    /// \brief Returns the value of the constant node as an AxisVector
    ///        object
    ///        Can only be used on element::i64 nodes and interprets
    ///        negative values as zeros.
    AxisVector get_axis_vector_val() const;
    /// \brief Returns the value of the constant node as an AxisSet
    ///        object
    ///        Can only be used on element::i64 nodes and interprets
    ///        negative values as zeros.
    ///        Repeated values are allowed.
    AxisSet get_axis_set_val() const;

    /// \brief Return data size in bytes
    size_t get_byte_size() const;

    /// \brief Wrapper around constructing a shared_ptr of a Constant
    ///
    /// \param type The element type of the tensor constant.
    /// \param shape The shape of the tensor constant.
    /// \param values A vector of values to use as the constant data.
    template <typename T>
    static std::shared_ptr<Constant> create(const element::Type& type,
                                            const Shape& shape,
                                            const std::vector<T>& values) {
        return std::make_shared<Constant>(type, shape, values);
    }

    /// \brief Wrapper around constructing a shared_ptr of a Constant
    ///
    /// \param type The element type of the tensor constant.
    /// \param shape The shape of the tensor constant.
    /// \param values An initializer_list of values to use as the constant data.
    template <typename T>
    static std::shared_ptr<Constant> create(const element::Type& type,
                                            const Shape& shape,
                                            std::initializer_list<T> values) {
        return std::make_shared<Constant>(type, shape, std::vector<T>{values});
    }

    /// \brief Wrapper around constructing a shared_ptr of a Constant
    ///
    /// \param type The element type of the tensor constant.
    /// \param shape The shape of the tensor constant.
    /// \param memory An continues memory chunk which contains the constant data.
    static std::shared_ptr<Constant> create(const element::Type& type, const Shape& shape, const void* memory) {
        return std::make_shared<Constant>(type, shape, memory);
    }

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    /// \return The initialization literals for the tensor constant.
    std::vector<std::string> get_value_strings() const;

    template <typename T>
    std::vector<T> get_vector() const {
        const T* p = get_data_ptr<T>();
        if (p == nullptr) {
            OPENVINO_THROW("Cannot create vector! Buffer is not allocated.");
        }
        return std::vector<T>(p, p + shape_size(m_shape));
    }

    /// \brief Return the Constant's value as a vector cast to type T
    ///
    /// \tparam T             Type to which data vector's entries will be cast.
    /// \param  num_elements  (Optional) Number of elements to cast. In default case returns all elements
    /// \return    Constant's data vector.
    template <typename T>
    std::vector<T> cast_vector(int64_t num_elements = -1) const {
        auto source_type = get_element_type();
        std::vector<T> rc;
        using Type_t = element::Type_t;
#if defined(_MSC_VER)
#    pragma warning(push)
#    pragma warning(disable : 4244)
#endif
        size_t num_elements_in_constant = shape_size(m_shape);
        size_t num_elements_to_cast =
            (num_elements < 0 ? num_elements_in_constant
                              : std::min(static_cast<size_t>(num_elements), num_elements_in_constant));
        switch (source_type) {
        case Type_t::boolean:
            cast_vector<Type_t::boolean>(rc, num_elements_to_cast);
            break;
        case Type_t::bf16:
            cast_vector<Type_t::bf16>(rc, num_elements_to_cast);
            break;
        case Type_t::f16:
            cast_vector<Type_t::f16>(rc, num_elements_to_cast);
            break;
        case Type_t::f32:
            cast_vector<Type_t::f32>(rc, num_elements_to_cast);
            break;
        case Type_t::f64:
            cast_vector<Type_t::f64>(rc, num_elements_to_cast);
            break;
        case Type_t::i4:
            cast_vector<Type_t::i4>(rc, num_elements_to_cast);
            break;
        case Type_t::i8:
            cast_vector<Type_t::i8>(rc, num_elements_to_cast);
            break;
        case Type_t::i16:
            cast_vector<Type_t::i16>(rc, num_elements_to_cast);
            break;
        case Type_t::i32:
            cast_vector<Type_t::i32>(rc, num_elements_to_cast);
            break;
        case Type_t::i64:
            cast_vector<Type_t::i64>(rc, num_elements_to_cast);
            break;
        case Type_t::u1:
            cast_vector<Type_t::u1>(rc, num_elements_to_cast);
            break;
        case Type_t::u4:
            cast_vector<Type_t::u4>(rc, num_elements_to_cast);
            break;
        case Type_t::u8:
            cast_vector<Type_t::u8>(rc, num_elements_to_cast);
            break;
        case Type_t::u16:
            cast_vector<Type_t::u16>(rc, num_elements_to_cast);
            break;
        case Type_t::u32:
            cast_vector<Type_t::u32>(rc, num_elements_to_cast);
            break;
        case Type_t::u64:
            cast_vector<Type_t::u64>(rc, num_elements_to_cast);
            break;
        case Type_t::string:
            cast_vector<Type_t::string>(rc, num_elements_to_cast);
            break;
        default:
            OPENVINO_THROW("unsupported type");
        }
#if defined(_MSC_VER)
#    pragma warning(pop)
#endif
        return rc;
    }

    const void* get_data_ptr() const;

    template <typename T>
    const T* get_data_ptr() const {
        OPENVINO_ASSERT(sizeof(T) <= m_element_type.size() || shape_size(m_shape) <= 0, "Buffer over-read");

        return static_cast<const T*>(get_data_ptr());
    }

    template <element::Type_t ET>
    const typename element_type_traits<ET>::value_type* get_data_ptr() const {
        OPENVINO_ASSERT(ET == get_element_type(), "get_data_ptr() called for incorrect element type.");
        return static_cast<const typename element_type_traits<ET>::value_type*>(get_data_ptr());
    }

    bool get_all_data_elements_bitwise_identical() const {
        if (!m_all_elements_bitwise_identical_checked) {
            update_identical_flags(true, are_all_data_elements_bitwise_identical());
        }
        return m_all_elements_bitwise_identical;
    }
    std::string convert_value_to_string(size_t index) const;

    /**
     * \brief Allows to avoid buffer allocation on the visit_attributes call
     */
    void alloc_buffer_on_visit_attributes(bool val) {
        m_alloc_buffer_on_visit_attributes = val;
    }

private:
    Constant(bool memset_allocation, const element::Type& type, const Shape& shape);

    OPENVINO_SUPPRESS_DEPRECATED_START
    std::shared_ptr<ov::AlignedBuffer> legacy_to_ov_aligned_buffer(
        const std::shared_ptr<ngraph::runtime::AlignedBuffer>& buffer);
    OPENVINO_SUPPRESS_DEPRECATED_END

    template <element::Type_t Type,
              typename StorageDataType = fundamental_type_for<Type>,
              typename std::enable_if<Type != element::Type_t::u1 && Type != element::Type_t::u4 &&
                                          Type != element::Type_t::i4 && Type != element::Type_t::nf4,
                                      bool>::type = true>
    StorageDataType get_element_value(size_t index) const {
        return get_data_ptr<Type>()[index];
    }

    template <element::Type_t Type,
              typename StorageDataType = fundamental_type_for<Type>,
              typename std::enable_if<Type == element::Type_t::u1, bool>::type = true>
    StorageDataType get_element_value(size_t index) const {
        return (get_data_ptr<uint8_t>()[index / 8] >> (7 - (index % 8))) & 1;
    }

    template <element::Type_t Type,
              typename StorageDataType = fundamental_type_for<Type>,
              typename std::enable_if<Type == element::Type_t::u4, bool>::type = true>
    StorageDataType get_element_value(size_t index) const {
        return (get_data_ptr<uint8_t>()[index / 2] >> (index % 2 ? 4 : 0)) & 0x0F;
    }

    template <element::Type_t Type,
              typename StorageDataType = fundamental_type_for<Type>,
              typename std::enable_if<Type == element::Type_t::nf4, bool>::type = true>
    StorageDataType get_element_value(size_t index) const {
        return (get_data_ptr<uint8_t>()[index / 2] >> (index % 2 ? 4 : 0)) & 0x0F;
    }

    template <element::Type_t Type,
              typename StorageDataType = fundamental_type_for<Type>,
              typename std::enable_if<Type == element::Type_t::i4, bool>::type = true>
    StorageDataType get_element_value(size_t index) const {
        const uint8_t i4data = (get_data_ptr<uint8_t>()[index / 2] >> (index % 2 ? 4 : 0)) & 0x0F;
        const bool is_negative_number = (i4data >> 3) & 0x01;
        const int8_t data = is_negative_number ? i4data | 0xF0 : i4data;
        return data;
    }

    template <element::Type_t Type,
              typename OUT_T,
              typename std::enable_if<Type != element::Type_t::u1 && Type != element::Type_t::u4 &&
                                          Type != element::Type_t::i4 && Type != element::Type_t::string,
                                      bool>::type = true>
    void cast_vector(std::vector<OUT_T>& output_vector, size_t num_elements) const {
        // this function is workaround for waring during windows building
        // build complains for vector creation based on iterators
        // which point on different type than destination vector::value_type
        using IN_T = fundamental_type_for<Type>;
        auto first = get_data_ptr<IN_T>();
        auto output_size = std::min(num_elements, shape_size(m_shape));
        output_vector.reserve(output_size);

        std::transform(first, first + output_size, std::back_inserter(output_vector), [](IN_T c) {
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
            if (!std::is_same<OUT_T, IN_T>::value) {
                OPENVINO_ASSERT(!std::numeric_limits<IN_T>::is_signed || std::numeric_limits<OUT_T>::lowest() <= c,
                                "Cannot cast vector from ",
                                Type,
                                " constant to ",
                                element::from<OUT_T>(),
                                ". Some values are outside the range. Example: ",
                                c);
                OPENVINO_ASSERT(std::numeric_limits<OUT_T>::max() >= c,
                                "Cannot cast vector from ",
                                Type,
                                " constant to ",
                                element::from<OUT_T>(),
                                ". Some values are outside the range. Example: ",
                                c);
            }
#if defined(__clang__)
#    pragma clang diagnostic pop
#elif defined(__GNUC__)
#    pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#    pragma warning(pop)
#endif
            return static_cast<OUT_T>(c);
        });
    }

    template <element::Type_t Type, typename std::enable_if<Type == element::Type_t::string, bool>::type = true>
    void cast_vector(std::vector<std::string>& output_vector, size_t num_elements) const {
        auto output_size = std::min(num_elements, shape_size(m_shape));
        output_vector.reserve(output_size);
        const auto p = get_data_ptr<Type>();
        std::copy(p, p + output_size, std::back_inserter(output_vector));
    }

    template <element::Type_t Type, typename std::enable_if<Type != element::Type_t::string, bool>::type = true>
    void cast_vector(std::vector<std::string>& output_vector, size_t num_elements) const {
        OPENVINO_THROW("cast_vector does not support casting ov::Tensor of type " +
                       ov::element::Type(Type).to_string() + "to std::vector of std::string elements");
    }

    template <element::Type_t Type,
              typename OUT_T,
              typename std::enable_if<Type == element::Type_t::string, bool>::type = true>
    void cast_vector(std::vector<OUT_T>& output_vector, size_t num_elements) const {
        auto output_type = std::string(typeid(OUT_T{}).name());
        OPENVINO_THROW("cast_vector does not support casting string ov::Tensor to std::vector with elements of type " +
                       output_type);
    }

    template <element::Type_t Type,
              typename OUT_T,
              typename std::enable_if<Type == element::Type_t::u1, bool>::type = true>
    void cast_vector(std::vector<OUT_T>& output, size_t num_elements) const {
        using IN_T = fundamental_type_for<Type>;
        const auto element_number = std::min(num_elements, shape_size(m_shape));
        const auto source_begin = get_data_ptr<uint8_t>();
        const auto source_end = std::next(source_begin, (element_number + 7) / 8);
        const auto round_element_no = element_number % 8 ? element_number - element_number % 8 + 8 : element_number;
        output.reserve(round_element_no);  // adds 7 more elements here?
        std::for_each(source_begin, source_end, [&](IN_T c) {
            for (const auto i : {7, 6, 5, 4, 3, 2, 1, 0}) {
                const uint8_t data = (c >> i) & 0x01;
                output.push_back(data);
            }
        });
        output.resize(element_number);
    }

    template <element::Type_t Type,
              typename OUT_T,
              typename std::enable_if<Type == element::Type_t::u4, bool>::type = true>
    void cast_vector(std::vector<OUT_T>& output, size_t num_elements) const {
        using IN_T = fundamental_type_for<Type>;
        const auto element_number = std::min(num_elements, shape_size(m_shape));
        const auto source_begin = get_data_ptr<uint8_t>();
        const auto source_end = std::next(source_begin, (element_number + 1) / 2);
        const auto round_element_no = element_number % 2 ? element_number + 1 : element_number;
        output.reserve(round_element_no);  // adds 1 more elements here?
        std::for_each(source_begin, source_end, [&](IN_T c) {
            for (const auto i : {0, 4}) {
                const uint8_t data = (c >> i) & 0x0F;
                output.push_back(data);
            }
        });
        output.resize(element_number);
    }
    template <element::Type_t Type,
              typename OUT_T,
              typename std::enable_if<Type == element::Type_t::i4, bool>::type = true>
    void cast_vector(std::vector<OUT_T>& output, size_t num_elements) const {
        using IN_T = fundamental_type_for<Type>;
        const auto element_number = std::min(num_elements, shape_size(m_shape));
        const auto source_begin = get_data_ptr<uint8_t>();
        const auto source_end = std::next(source_begin, (element_number + 1) / 2);
        const auto round_element_no = element_number % 2 ? element_number + 1 : element_number;
        output.reserve(round_element_no);  // adds 1 more elements here?
        std::for_each(source_begin, source_end, [&](IN_T c) {
            for (const auto i : {0, 4}) {
                const uint8_t i4data = (c >> i) & 0x0F;
                const bool is_negative_number = (i4data >> 3) & 0x01;
                const int8_t data = is_negative_number ? i4data | 0xF0 : i4data;
                output.push_back(data);
            }
        });
        output.resize(element_number);
    }

    template <element::Type_t Type,
              typename StorageDataType = fundamental_type_for<Type>,
              typename std::enable_if<Type != element::Type_t::string, bool>::type = true>
    void fill_data(const std::string& value) {
        OPENVINO_THROW("Called fill_data(std::string) with non-string element_type");
    }

    template <element::Type_t Type,
              typename T,
              typename StorageDataType = fundamental_type_for<Type>,
              typename std::enable_if<Type != element::Type_t::u1 && Type != element::Type_t::u4 &&
                                          Type != element::Type_t::i4 && Type != element::Type_t::nf4 &&
                                          Type != element::Type_t::string,
                                      bool>::type = true>
    void fill_data(const T& value) {
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
        if (!std::is_same<T, StorageDataType>::value) {
            OPENVINO_ASSERT(
                !std::numeric_limits<T>::is_signed || std::numeric_limits<StorageDataType>::lowest() <= value,
                "Cannot fill constant data. Values is outside the range.");
            OPENVINO_ASSERT(std::numeric_limits<StorageDataType>::max() >= value,
                            "Cannot fill constant data. Values is outside the range.");
        }
#if defined(__clang__)
#    pragma clang diagnostic pop
#elif defined(__GNUC__)
#    pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#    pragma warning(pop)
#endif

        const auto size = shape_size(m_shape);
        const auto v = static_cast<StorageDataType>(value);
        std::fill_n(get_data_ptr_nc<Type>(), size, v);
    }

    template <element::Type_t Type, typename std::enable_if<Type == element::Type_t::string, bool>::type = true>
    void fill_data(const std::string& value) {
        auto num_elements = shape_size(m_shape);
        std::uninitialized_fill_n(get_data_ptr_nc<Type>(), num_elements, value);
    }

    template <element::Type_t Type,
              typename T,
              typename StorageDataType = fundamental_type_for<Type>,
              typename std::enable_if<Type == element::Type_t::string, bool>::type = true>
    void fill_data(const T& value) {
        std::string type_name(typeid(value).name());
        OPENVINO_THROW("fill_data does not support to fill ov::Tensor of string type with value of " + type_name);
    }

    template <element::Type_t Type,
              typename T,
              typename StorageDataType = fundamental_type_for<Type>,
              typename std::enable_if<Type == element::Type_t::u1, bool>::type = true>
    void fill_data(const T& value) {
        const StorageDataType v = value ? 0xFF : 0x00;
        std::fill_n(get_data_ptr_nc<Type>(), mem_size(), v);
    }

    template <element::Type_t Type,
              typename T,
              typename StorageDataType = fundamental_type_for<Type>,
              typename std::enable_if<Type == element::Type_t::u4 || Type == element::Type_t::i4 ||
                                          Type == element::Type_t::nf4,
                                      bool>::type = true>
    void fill_data(const T& value) {
        uint8_t v = value_in_range<Type>(value);
        v &= 0x0F;
        v += v << 4;
        std::fill_n(get_data_ptr_nc<Type>(), mem_size(), v);
    }

    void allocate_buffer(bool memset_allocation);

    void* get_data_ptr_nc();

    template <element::Type_t ET>
    typename element_type_traits<ET>::value_type* get_data_ptr_nc() {
        OPENVINO_ASSERT(ET == get_element_type(), "get_data_ptr_nc() called for incorrect element type.");
        return static_cast<typename element_type_traits<ET>::value_type*>(get_data_ptr_nc());
    }

    Constant(const OutputVector& args) : Op(args), m_shape({}) {}

    virtual void infer_element_type() {}
    template <typename T>
    void write_values(const std::vector<T>& values) {
        write_to_buffer(values);
    }

    template <element::Type_t Type,
              typename T,
              typename StorageDataType = fundamental_type_for<Type>,
              typename std::enable_if<Type != element::Type_t::nf4 && Type != element::Type_t::u1 &&
                                          Type != element::Type_t::u4 && Type != element::Type_t::i4 &&
                                          Type != element::Type_t::string,
                                      bool>::type = true>
    void write_buffer(const std::vector<T>& source) {
        auto p = get_data_ptr_nc<Type>();
        for (size_t i = 0; i < source.size(); i++) {
            p[i] = static_cast<StorageDataType>(source[i]);
        }
    }

    template <element::Type_t Type, typename std::enable_if<Type == element::Type_t::string, bool>::type = true>
    void write_buffer(const std::vector<std::string>& source) {
        // elements of string ov::Tensor is already pre-initialized in allocate_buffer
        auto p = get_data_ptr_nc<Type>();
        auto num_elements = std::min(shape_size(m_shape), source.size());
        std::uninitialized_copy_n(source.begin(), num_elements, p);
    }

    template <element::Type_t Type, typename std::enable_if<Type != element::Type_t::string, bool>::type = true>
    void write_buffer(const std::vector<std::string>& source) {
        OPENVINO_THROW("write_buffer does not support writing std::string elements into ov::Tensor of type:" +
                       ov::element::Type(Type).to_string());
    }

    template <element::Type_t Type,
              typename T,
              typename std::enable_if<Type == element::Type_t::string, bool>::type = true>
    void write_buffer(const std::vector<T>& source) {
        if (source.size() > 0) {
            auto source_type = std::string(typeid(source[0]).name());
            OPENVINO_THROW("write_buffer does not support writing elements of type " + source_type +
                           " into string ov::Tensor");
        }
    }

    template <element::Type_t Type,
              typename T,
              typename StorageDataType = fundamental_type_for<Type>,
              typename std::enable_if<Type == element::Type_t::u4 || Type == element::Type_t::i4 ||
                                          (Type == element::Type_t::nf4 && std::is_integral<T>::value),
                                      bool>::type = true>
    void write_buffer(const std::vector<T>& source) {
        auto p = get_data_ptr_nc<Type>();
        size_t i = 0;
        for (; i < source.size() / 2; i++) {
            const auto v1 = value_in_range<Type>(source[i * 2]) & 0x0F;
            const auto v2 = value_in_range<Type>(source[i * 2 + 1]) & 0x0F;
            const auto v = (v2 << 4) | v1;
            p[i] = static_cast<StorageDataType>(v);
        }
        if (source.size() % 2) {
            const auto v = value_in_range<Type>(source[i * 2]) & 0x0F;
            p[i] = static_cast<StorageDataType>(v);
        }
    }

    template <element::Type_t Type,
              typename T,
              typename StorageDataType = fundamental_type_for<Type>,
              typename std::enable_if<Type == element::Type_t::nf4 &&
                                          (std::is_floating_point<T>::value || std::is_same<T, bfloat16>::value ||
                                           std::is_same<T, float16>::value),
                                      bool>::type = true>
    void write_buffer(const std::vector<T>& source) {
        auto p = get_data_ptr_nc<Type>();
        size_t i = 0;
        for (; i < source.size() / 2; i++) {
            const auto idx1 = quantize_nf4(static_cast<float>(source[i * 2]));
            const auto idx2 = quantize_nf4(static_cast<float>(source[i * 2 + 1]));
            const auto v1 = value_in_range<Type>(idx1) & 0x0F;
            const auto v2 = value_in_range<Type>(idx2) & 0x0F;
            const auto v = (v2 << 4) | v1;
            p[i] = static_cast<StorageDataType>(v);
        }
        if (source.size() % 2) {
            const auto idx1 = quantize_nf4(static_cast<float>(source[i * 2]));
            const auto v = value_in_range<Type>(idx1) & 0x0F;
            p[i] = static_cast<StorageDataType>(v);
        }
    }

    template <element::Type_t Type,
              typename T,
              typename StorageDataType = fundamental_type_for<Type>,
              typename std::enable_if<Type == element::Type_t::u1, bool>::type = true>
    void write_buffer(const std::vector<T>& source) {
        auto p = get_data_ptr_nc<Type>();
        size_t i = 0;
        for (; i < source.size() / 8; i++) {
            uint8_t v{};
            for (int j = 0; j != 8; j++) {
                const uint8_t b = source[i * 8 + j] ? 0x01 << (7 - j) : 0;
                v |= b;
            }
            p[i] = static_cast<StorageDataType>(v);
        }
        uint8_t v{};
        for (unsigned j = 0; j != source.size() % 8; j++) {
            const uint8_t b = source[i * 8 + j] ? 0x01 << (7 - j) : 0;
            v |= b;
        }
        p[i] = static_cast<StorageDataType>(v);
    }

    template <typename T>
    void write_to_buffer(const std::vector<T>& source) {
        const auto& target_type = m_element_type;
        size_t target_element_count = shape_size(m_shape);
        if (source.size() != target_element_count) {
            OPENVINO_THROW("Constant initializer does not match shape");
        }
        using Type_t = element::Type_t;
#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic error "-Wswitch"
#    pragma GCC diagnostic error "-Wswitch-enum"
#endif
        switch (target_type) {
        case Type_t::boolean:
            write_buffer<Type_t::boolean>(source);
            break;
        case Type_t::bf16:
            write_buffer<Type_t::bf16>(source);
            break;
        case Type_t::f16:
            write_buffer<Type_t::f16>(source);
            break;
        case Type_t::f32:
            write_buffer<Type_t::f32>(source);
            break;
        case Type_t::f64:
            write_buffer<Type_t::f64>(source);
            break;
        case Type_t::i4:
            write_buffer<Type_t::i4>(source);
            break;
        case Type_t::i8:
            write_buffer<Type_t::i8>(source);
            break;
        case Type_t::i16:
            write_buffer<Type_t::i16>(source);
            break;
        case Type_t::i32:
            write_buffer<Type_t::i32>(source);
            break;
        case Type_t::i64:
            write_buffer<Type_t::i64>(source);
            break;
        case Type_t::u1:
            write_buffer<Type_t::u1>(source);
            break;
        case Type_t::u4:
            write_buffer<Type_t::u4>(source);
            break;
        case Type_t::u8:
            write_buffer<Type_t::u8>(source);
            break;
        case Type_t::u16:
            write_buffer<Type_t::u16>(source);
            break;
        case Type_t::u32:
            write_buffer<Type_t::u32>(source);
            break;
        case Type_t::u64:
            write_buffer<Type_t::u64>(source);
            break;
        case Type_t::nf4:
            write_buffer<Type_t::nf4>(source);
            break;
        case Type_t::string:
            write_buffer<Type_t::string>(source);
            break;
        case element::Type_t::undefined:
        case element::Type_t::dynamic:
            OPENVINO_THROW("unsupported type");
        }
#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#    pragma GCC diagnostic pop
#endif
    }
    template <ov::element::Type_t Type,
              typename ValueT,
              typename std::enable_if<Type == ov::element::Type_t::u4 || Type == ov::element::Type_t::u4 ||
                                          Type == ov::element::Type_t::nf4,
                                      bool>::type = true>
    static ov::fundamental_type_for<Type> value_in_range(const ValueT& value) {
        const auto result = static_cast<ov::fundamental_type_for<Type>>(value);
        OPENVINO_ASSERT(0 <= result && result <= 15, "assigned value out of range u4 values");
        return result;
    }

    template <ov::element::Type_t Type,
              typename ValueT,
              typename std::enable_if<Type == ov::element::Type_t::i4, bool>::type = true>
    static ov::fundamental_type_for<Type> value_in_range(const ValueT& value) {
        const auto result = ov::fundamental_type_for<Type>(value);
        OPENVINO_ASSERT(-8 <= result && result <= 7, "assigned value out of range i4 values");
        return result;
    }

    bool are_all_data_elements_bitwise_identical() const;
    // This is 'const' as it updates only mutable data
    void update_identical_flags(bool is_checked, bool identical_value) const;
    static constexpr size_t host_alignment() {
        return 64;
    }

    size_t mem_size() const {
        constexpr size_t bits_in_byte = 8;
        const auto bit_width = m_element_type.bitwidth();
        auto size = shape_size(m_shape);
        if (bit_width < bits_in_byte) {
            size *= bit_width;
            return (size % bits_in_byte) ? (size / bits_in_byte) + 1 : (size / bits_in_byte);
        } else {
            return size * m_element_type.size();
        }
    }

    static uint8_t quantize_nf4(float x);

    friend struct ValueToString;

    element::Type m_element_type;
    Shape m_shape{};
    std::shared_ptr<ov::AlignedBuffer> m_data;
    mutable std::atomic_bool m_all_elements_bitwise_identical{false};
    mutable std::atomic_bool m_all_elements_bitwise_identical_checked{false};
    bool m_alloc_buffer_on_visit_attributes = true;
};
}  // namespace v0
}  // namespace op
}  // namespace ov
