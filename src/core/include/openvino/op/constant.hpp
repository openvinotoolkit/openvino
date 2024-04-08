// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstring>

#include "openvino/core/axis_set.hpp"
#include "openvino/core/axis_vector.hpp"
#include "openvino/core/coordinate_diff.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rtti.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/element_type_traits.hpp"
#include "openvino/op/op.hpp"

namespace ov {

class AlignedBuffer;

namespace element {
template <Type_t ET, class T>
class Iterator;
}

namespace op {
namespace v0 {
/// \brief Class for constants.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Constant : public Op {
public:
    OPENVINO_OP("Constant", "opset1");

    Constant() = default;

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

        fill_or_write(has_single_value, type, values);
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
            fill_lp_data<Type_t::i4>(value);
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
            fill_lp_data<Type_t::u1>(value);
            break;
        case Type_t::u4:
            fill_lp_data<Type_t::u4>(value);
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
            fill_lp_data<Type_t::nf4>(value);
            break;
        case Type_t::f8e4m3:
            fill_data<Type_t::f8e4m3>(value);
            break;
        case Type_t::f8e5m2:
            fill_data<Type_t::f8e5m2>(value);
            break;
        case Type_t::string:
            fill_data<Type_t::string>(value);
            break;
        case Type_t::u2:
        case Type_t::u3:
        case Type_t::u6:
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

    Constant(const element::Type& type, const Shape& shape, const std::shared_ptr<ov::AlignedBuffer>& data);

    Constant(const Constant& other);
    Constant(const Constant& other, const Shape& new_shape);
    Constant& operator=(const Constant&) = delete;

    ~Constant() override;

    void validate_and_infer_types() override;

    bool visit_attributes(AttributeVisitor& visitor) override;

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;
    bool has_evaluate() const override;
    bool evaluate_lower(TensorVector& outputs) const override;
    bool evaluate_upper(TensorVector& outputs) const override;

    // Don't constant fold a constant; it would make a copy
    bool constant_fold(OutputVector& outputs, const OutputVector& inputs) override;

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
        std::vector<T> rc;
        using Type_t = element::Type_t;

        const auto num_elements_in_constant = shape_size(m_shape);
        const auto num_elements_to_cast =
            (num_elements < 0 ? num_elements_in_constant
                              : std::min(static_cast<size_t>(num_elements), num_elements_in_constant));
        rc.reserve(num_elements_to_cast);

        switch (m_element_type) {
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
            cast_lp_vector<Type_t::i4>(rc, num_elements_to_cast);
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
            cast_lp_vector<Type_t::u1>(rc, num_elements_to_cast);
            break;
        case Type_t::u4:
            cast_lp_vector<Type_t::u4>(rc, num_elements_to_cast);
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
        case Type_t::f8e4m3:
            cast_vector<Type_t::f8e4m3>(rc, num_elements_to_cast);
            break;
        case Type_t::f8e5m2:
            cast_vector<Type_t::f8e5m2>(rc, num_elements_to_cast);
            break;
        case Type_t::string:
            cast_vector<Type_t::string>(rc, num_elements_to_cast);
            break;
        default:
            OPENVINO_THROW("unsupported type");
        }
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

    bool get_all_data_elements_bitwise_identical() const;

    std::string convert_value_to_string(size_t index) const;

    /**
     * \brief Allows to avoid buffer allocation on the visit_attributes call
     */
    void alloc_buffer_on_visit_attributes(bool val);

private:
    Constant(bool memset_allocation, const element::Type& type, const Shape& shape);

    template <
        element::Type_t Type,
        class OUT_T,
        typename std::enable_if<Type != element::string && !std::is_same<OUT_T, std::string>::value>::type* = nullptr>
    void cast_vector(std::vector<OUT_T>& output_vector, size_t num_elements) const {
        // this function is workaround for waring during windows building
        // build complains for vector creation based on iterators
        // which point on different type than destination vector::value_type
        using IN_T = fundamental_type_for<Type>;
        auto first = get_data_ptr<IN_T>();
        std::transform(first, first + num_elements, std::back_inserter(output_vector), [](IN_T c) {
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

    template <element::Type_t Type,
              class U,
              typename std::enable_if<Type == element::string && std::is_same<U, std::string>::value>::type* = nullptr>
    void cast_vector(std::vector<U>& output_vector, size_t num_elements) const {
        const auto p = get_data_ptr<Type>();
        std::copy_n(p, num_elements, std::back_inserter(output_vector));
    }

    template <
        element::Type_t Type,
        class U,
        typename std::enable_if<(Type == element::string) != std::is_same<U, std::string>::value>::type* = nullptr>
    void cast_vector(std::vector<U>& output, size_t num_elements) const {
        OPENVINO_THROW("'cast_vector' does not support casting Constant of type ",
                       Type,
                       " into std::vector of ",
                       element::from<U>());
    }

    // generic cast_LP_data if input is not std or OV type (do additional conversion)
    template <element::Type_t ET, class U>
    void cast_lp_vector(std::vector<U>& output, size_t num_elements) const {
        auto lp_buffer = LPBuffer<ET>(get_data_ptr());
        auto out_inserter = std::back_inserter(output);
        for (size_t i = 0; i < num_elements; ++i, ++lp_buffer) {
            *out_inserter = lp_buffer.read();
        }
    }

    template <element::Type_t ET>
    void cast_lp_vector(std::vector<std::string>& output, size_t num_elements) const {
        cast_vector<element::i8>(output, num_elements);
    }

    template <element::Type_t Type,
              class T,
              typename std::enable_if<Type != element::string && !std::is_same<T, std::string>::value>::type* = nullptr>
    void fill_data(const T& value) {
        using StorageDataType = ov::fundamental_type_for<Type>;
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

    template <element::Type_t Type,
              class T,
              typename std::enable_if<Type == element::string && std::is_same<T, std::string>::value>::type* = nullptr>
    void fill_data(const T& value) {
        auto num_elements = shape_size(m_shape);
        std::uninitialized_fill_n(get_data_ptr_nc<Type>(), num_elements, value);
    }

    template <
        element::Type_t Type,
        class T,
        typename std::enable_if<(Type == element::string) != std::is_same<T, std::string>::value>::type* = nullptr>
    void fill_data(const T& value) {
        if (Type == element::string) {
            fill_data<element::string, std::string>(std::string());
        }
        OPENVINO_THROW("'fill_data' does not support writing elements of type ",
                       element::from<T>(),
                       " into Constant of type ",
                       Type);
    }

    // generic fill_lp_data if input is not std or OV type (do additional conversion)
    template <element::Type_t ET, class T>
    void fill_lp_data(const T& value) {
        fill_lp_data<ET>(static_cast<float>(value));
    }

    template <element::Type_t ET>
    void fill_lp_data(const std::string& value) {
        fill_data<element::i8>(value);
    }

    void allocate_buffer(bool memset_allocation);

    void* get_data_ptr_nc();

    template <element::Type_t ET>
    typename ov::fundamental_type_for<ET>* get_data_ptr_nc() {
        OPENVINO_ASSERT(ET == get_element_type(), "get_data_ptr_nc() called for incorrect element type.");
        return static_cast<typename element_type_traits<ET>::value_type*>(get_data_ptr_nc());
    }

    template <typename T>
    void write_values(const std::vector<T>& values) {
        write_to_buffer(values);
    }

    template <element::Type_t Type,
              typename T,
              typename std::enable_if<Type != element::string && !std::is_same<T, std::string>::value>::type* = nullptr>
    void write_buffer(const std::vector<T>& source) {
        using StorageDataType = fundamental_type_for<Type>;
        auto p = get_data_ptr_nc<Type>();
        for (size_t i = 0; i < source.size(); ++i) {
            p[i] = static_cast<StorageDataType>(source[i]);
        }
    }

    template <element::Type_t Type,
              typename T,
              typename std::enable_if<Type == element::string && std::is_same<T, std::string>::value>::type* = nullptr>
    void write_buffer(const std::vector<T>& source) {
        // elements of string are already pre-initialized in allocate_buffer
        auto p = get_data_ptr_nc<Type>();
        std::uninitialized_copy_n(source.begin(), source.size(), p);
    }

    template <
        element::Type_t Type,
        typename T,
        typename std::enable_if<(Type == element::string) != std::is_same<T, std::string>::value>::type* = nullptr>
    void write_buffer(const std::vector<T>& source) {
        if (Type == element::string) {
            fill_data<element::string>(std::string());
        }
        OPENVINO_THROW("'write_buffer' does not support writing elements of type ",
                       element::from<T>(),
                       " into Constant of type ",
                       Type);
    }

    // generic write_lp_buffer if input is not std or OV type (do additional conversion)
    template <element::Type_t ET, class T>
    void write_lp_buffer(const std::vector<T>& source) {
        auto lp_buffer = LPBuffer<ET>(get_data_ptr_nc());
        for (const auto& value : source) {
            lp_buffer.write(static_cast<float>(value));
            ++lp_buffer;
        }
    }

    template <element::Type_t ET>
    void write_lp_buffer(const std::vector<std::string>& source) {
        write_buffer<element::i8>(source);
    }

    template <typename T>
    void write_to_buffer(const std::vector<T>& source) {
        if (source.size() != shape_size(m_shape)) {
            OPENVINO_THROW("Constant initializer does not match shape");
        }
        using Type_t = element::Type_t;
#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic error "-Wswitch"
#    pragma GCC diagnostic error "-Wswitch-enum"
#endif
        switch (m_element_type) {
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
            write_lp_buffer<Type_t::i4>(source);
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
            write_lp_buffer<Type_t::u1>(source);
            break;
        case Type_t::u4:
            write_lp_buffer<Type_t::u4>(source);
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
            write_lp_buffer<Type_t::nf4>(source);
            break;
        case Type_t::f8e4m3:
            write_buffer<Type_t::f8e4m3>(source);
            break;
        case Type_t::f8e5m2:
            write_buffer<Type_t::f8e5m2>(source);
            break;
        case Type_t::string:
            write_buffer<Type_t::string>(source);
            break;
        case element::Type_t::u2:
        case element::Type_t::u3:
        case element::Type_t::u6:
        case element::Type_t::undefined:
        case element::Type_t::dynamic:
            OPENVINO_THROW("unsupported type");
        }
#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#    pragma GCC diagnostic pop
#endif
    }

    template <class T>
    void fill_or_write(const bool fill, const element::Type& et, const std::vector<T>& values) {
        if (fill) {
            fill_data<T>(et, values[0]);
        } else {
            write_values(values);
        }
    }

    bool are_all_data_elements_bitwise_identical() const;
    // This is 'const' as it updates only mutable data
    void update_identical_flags(bool is_checked, bool identical_value) const;

    static constexpr size_t host_alignment() {
        return 64;
    }

    // Internal helper to read/write low precision values for not standard or OV type.
    template <element::Type_t ET>
    struct LPBuffer {
        using lp_iter = element::Iterator<ET, typename ov::fundamental_type_for<ET>>;
        using lp_iter_ptr = std::shared_ptr<lp_iter>;

        LPBuffer(void* ptr);
        LPBuffer(const void* ptr) : LPBuffer{const_cast<void*>(ptr)} {}
        void write(const float value);
        ov::fundamental_type_for<ET> read() const;
        LPBuffer& operator++();

        lp_iter_ptr iter;
    };

    element::Type m_element_type{};
    Shape m_shape{};
    std::shared_ptr<ov::AlignedBuffer> m_data{};
    mutable std::atomic_bool m_all_elements_bitwise_identical{false};
    mutable std::atomic_bool m_all_elements_bitwise_identical_checked{false};
    bool m_alloc_buffer_on_visit_attributes{true};
};

template <>
OPENVINO_API Constant::LPBuffer<element::u1>::LPBuffer(void* ptr);
template <>
OPENVINO_API Constant::LPBuffer<element::u4>::LPBuffer(void* ptr);
template <>
OPENVINO_API Constant::LPBuffer<element::i4>::LPBuffer(void* ptr);
template <>
OPENVINO_API Constant::LPBuffer<element::nf4>::LPBuffer(void* ptr);

template <>
OPENVINO_API void Constant::LPBuffer<element::u1>::write(const float value);
template <>
OPENVINO_API void Constant::LPBuffer<element::u4>::write(const float value);
template <>
OPENVINO_API void Constant::LPBuffer<element::i4>::write(const float value);
template <>
OPENVINO_API void Constant::LPBuffer<element::nf4>::write(const float value);

template <>
OPENVINO_API ov::fundamental_type_for<element::u1> Constant::LPBuffer<element::u1>::read() const;
template <>
OPENVINO_API ov::fundamental_type_for<element::u4> Constant::LPBuffer<element::u4>::read() const;
template <>
OPENVINO_API ov::fundamental_type_for<element::i4> Constant::LPBuffer<element::i4>::read() const;
template <>
OPENVINO_API ov::fundamental_type_for<element::nf4> Constant::LPBuffer<element::nf4>::read() const;

template <>
OPENVINO_API Constant::LPBuffer<element::u1>& Constant::LPBuffer<element::u1>::operator++();
template <>
OPENVINO_API Constant::LPBuffer<element::u4>& Constant::LPBuffer<element::u4>::operator++();
template <>
OPENVINO_API Constant::LPBuffer<element::i4>& Constant::LPBuffer<element::i4>::operator++();
template <>
OPENVINO_API Constant::LPBuffer<element::nf4>& Constant::LPBuffer<element::nf4>::operator++();

#define CONSTANT_FILL_DATA_SPECIALIZATION(ET, SRC_TYPE) \
    template <>                                         \
    OPENVINO_API void Constant::fill_lp_data<element::Type_t::ET>(const SRC_TYPE& value);

CONSTANT_FILL_DATA_SPECIALIZATION(u1, bool)
CONSTANT_FILL_DATA_SPECIALIZATION(u1, char)
CONSTANT_FILL_DATA_SPECIALIZATION(u1, signed char)
CONSTANT_FILL_DATA_SPECIALIZATION(u1, unsigned char)
CONSTANT_FILL_DATA_SPECIALIZATION(u1, short)
CONSTANT_FILL_DATA_SPECIALIZATION(u1, unsigned short)
CONSTANT_FILL_DATA_SPECIALIZATION(u1, int)
CONSTANT_FILL_DATA_SPECIALIZATION(u1, unsigned int)
CONSTANT_FILL_DATA_SPECIALIZATION(u1, long)
CONSTANT_FILL_DATA_SPECIALIZATION(u1, unsigned long)
CONSTANT_FILL_DATA_SPECIALIZATION(u1, long long)
CONSTANT_FILL_DATA_SPECIALIZATION(u1, unsigned long long)
CONSTANT_FILL_DATA_SPECIALIZATION(u1, float8_e4m3)
CONSTANT_FILL_DATA_SPECIALIZATION(u1, float8_e5m2)
CONSTANT_FILL_DATA_SPECIALIZATION(u1, float16)
CONSTANT_FILL_DATA_SPECIALIZATION(u1, bfloat16)
CONSTANT_FILL_DATA_SPECIALIZATION(u1, float)
CONSTANT_FILL_DATA_SPECIALIZATION(u1, double)

CONSTANT_FILL_DATA_SPECIALIZATION(u4, bool)
CONSTANT_FILL_DATA_SPECIALIZATION(u4, char)
CONSTANT_FILL_DATA_SPECIALIZATION(u4, signed char)
CONSTANT_FILL_DATA_SPECIALIZATION(u4, unsigned char)
CONSTANT_FILL_DATA_SPECIALIZATION(u4, short)
CONSTANT_FILL_DATA_SPECIALIZATION(u4, unsigned short)
CONSTANT_FILL_DATA_SPECIALIZATION(u4, int)
CONSTANT_FILL_DATA_SPECIALIZATION(u4, unsigned int)
CONSTANT_FILL_DATA_SPECIALIZATION(u4, long)
CONSTANT_FILL_DATA_SPECIALIZATION(u4, unsigned long)
CONSTANT_FILL_DATA_SPECIALIZATION(u4, long long)
CONSTANT_FILL_DATA_SPECIALIZATION(u4, unsigned long long)
CONSTANT_FILL_DATA_SPECIALIZATION(u4, float8_e4m3)
CONSTANT_FILL_DATA_SPECIALIZATION(u4, float8_e5m2)
CONSTANT_FILL_DATA_SPECIALIZATION(u4, float16)
CONSTANT_FILL_DATA_SPECIALIZATION(u4, bfloat16)
CONSTANT_FILL_DATA_SPECIALIZATION(u4, float)
CONSTANT_FILL_DATA_SPECIALIZATION(u4, double)

CONSTANT_FILL_DATA_SPECIALIZATION(i4, bool)
CONSTANT_FILL_DATA_SPECIALIZATION(i4, char)
CONSTANT_FILL_DATA_SPECIALIZATION(i4, signed char)
CONSTANT_FILL_DATA_SPECIALIZATION(i4, unsigned char)
CONSTANT_FILL_DATA_SPECIALIZATION(i4, short)
CONSTANT_FILL_DATA_SPECIALIZATION(i4, unsigned short)
CONSTANT_FILL_DATA_SPECIALIZATION(i4, int)
CONSTANT_FILL_DATA_SPECIALIZATION(i4, unsigned int)
CONSTANT_FILL_DATA_SPECIALIZATION(i4, long)
CONSTANT_FILL_DATA_SPECIALIZATION(i4, unsigned long)
CONSTANT_FILL_DATA_SPECIALIZATION(i4, long long)
CONSTANT_FILL_DATA_SPECIALIZATION(i4, unsigned long long)
CONSTANT_FILL_DATA_SPECIALIZATION(i4, float8_e4m3)
CONSTANT_FILL_DATA_SPECIALIZATION(i4, float8_e5m2)
CONSTANT_FILL_DATA_SPECIALIZATION(i4, float16)
CONSTANT_FILL_DATA_SPECIALIZATION(i4, bfloat16)
CONSTANT_FILL_DATA_SPECIALIZATION(i4, float)
CONSTANT_FILL_DATA_SPECIALIZATION(i4, double)

CONSTANT_FILL_DATA_SPECIALIZATION(nf4, bool)
CONSTANT_FILL_DATA_SPECIALIZATION(nf4, char)
CONSTANT_FILL_DATA_SPECIALIZATION(nf4, signed char)
CONSTANT_FILL_DATA_SPECIALIZATION(nf4, unsigned char)
CONSTANT_FILL_DATA_SPECIALIZATION(nf4, short)
CONSTANT_FILL_DATA_SPECIALIZATION(nf4, unsigned short)
CONSTANT_FILL_DATA_SPECIALIZATION(nf4, int)
CONSTANT_FILL_DATA_SPECIALIZATION(nf4, unsigned int)
CONSTANT_FILL_DATA_SPECIALIZATION(nf4, long)
CONSTANT_FILL_DATA_SPECIALIZATION(nf4, unsigned long)
CONSTANT_FILL_DATA_SPECIALIZATION(nf4, long long)
CONSTANT_FILL_DATA_SPECIALIZATION(nf4, unsigned long long)
CONSTANT_FILL_DATA_SPECIALIZATION(nf4, float8_e4m3)
CONSTANT_FILL_DATA_SPECIALIZATION(nf4, float8_e5m2)
CONSTANT_FILL_DATA_SPECIALIZATION(nf4, float16)
CONSTANT_FILL_DATA_SPECIALIZATION(nf4, bfloat16)
CONSTANT_FILL_DATA_SPECIALIZATION(nf4, float)
CONSTANT_FILL_DATA_SPECIALIZATION(nf4, double)

#undef CONSTANT_FILL_DATA_SPECIALIZATION

#define CONSTANT_CAST_VECTOR_SPECIALIZATION(ET, DST_TYPE)                                                  \
    template <>                                                                                            \
    OPENVINO_API void Constant::cast_lp_vector<element::Type_t::ET>(std::vector<DST_TYPE> & output_vector, \
                                                                    size_t num_elements) const;

CONSTANT_CAST_VECTOR_SPECIALIZATION(u1, bool)
CONSTANT_CAST_VECTOR_SPECIALIZATION(u1, char)
CONSTANT_CAST_VECTOR_SPECIALIZATION(u1, signed char)
CONSTANT_CAST_VECTOR_SPECIALIZATION(u1, unsigned char)
CONSTANT_CAST_VECTOR_SPECIALIZATION(u1, short)
CONSTANT_CAST_VECTOR_SPECIALIZATION(u1, unsigned short)
CONSTANT_CAST_VECTOR_SPECIALIZATION(u1, int)
CONSTANT_CAST_VECTOR_SPECIALIZATION(u1, unsigned int)
CONSTANT_CAST_VECTOR_SPECIALIZATION(u1, long)
CONSTANT_CAST_VECTOR_SPECIALIZATION(u1, unsigned long)
CONSTANT_CAST_VECTOR_SPECIALIZATION(u1, long long)
CONSTANT_CAST_VECTOR_SPECIALIZATION(u1, unsigned long long)
CONSTANT_CAST_VECTOR_SPECIALIZATION(u1, float16)
CONSTANT_CAST_VECTOR_SPECIALIZATION(u1, bfloat16)
CONSTANT_CAST_VECTOR_SPECIALIZATION(u1, float)
CONSTANT_CAST_VECTOR_SPECIALIZATION(u1, double)

CONSTANT_CAST_VECTOR_SPECIALIZATION(u4, bool)
CONSTANT_CAST_VECTOR_SPECIALIZATION(u4, char)
CONSTANT_CAST_VECTOR_SPECIALIZATION(u4, signed char)
CONSTANT_CAST_VECTOR_SPECIALIZATION(u4, unsigned char)
CONSTANT_CAST_VECTOR_SPECIALIZATION(u4, short)
CONSTANT_CAST_VECTOR_SPECIALIZATION(u4, unsigned short)
CONSTANT_CAST_VECTOR_SPECIALIZATION(u4, int)
CONSTANT_CAST_VECTOR_SPECIALIZATION(u4, unsigned int)
CONSTANT_CAST_VECTOR_SPECIALIZATION(u4, long)
CONSTANT_CAST_VECTOR_SPECIALIZATION(u4, unsigned long)
CONSTANT_CAST_VECTOR_SPECIALIZATION(u4, long long)
CONSTANT_CAST_VECTOR_SPECIALIZATION(u4, unsigned long long)
CONSTANT_CAST_VECTOR_SPECIALIZATION(u4, float16)
CONSTANT_CAST_VECTOR_SPECIALIZATION(u4, bfloat16)
CONSTANT_CAST_VECTOR_SPECIALIZATION(u4, float)
CONSTANT_CAST_VECTOR_SPECIALIZATION(u4, double)

CONSTANT_CAST_VECTOR_SPECIALIZATION(i4, bool)
CONSTANT_CAST_VECTOR_SPECIALIZATION(i4, char)
CONSTANT_CAST_VECTOR_SPECIALIZATION(i4, signed char)
CONSTANT_CAST_VECTOR_SPECIALIZATION(i4, unsigned char)
CONSTANT_CAST_VECTOR_SPECIALIZATION(i4, short)
CONSTANT_CAST_VECTOR_SPECIALIZATION(i4, unsigned short)
CONSTANT_CAST_VECTOR_SPECIALIZATION(i4, int)
CONSTANT_CAST_VECTOR_SPECIALIZATION(i4, unsigned int)
CONSTANT_CAST_VECTOR_SPECIALIZATION(i4, long)
CONSTANT_CAST_VECTOR_SPECIALIZATION(i4, unsigned long)
CONSTANT_CAST_VECTOR_SPECIALIZATION(i4, long long)
CONSTANT_CAST_VECTOR_SPECIALIZATION(i4, unsigned long long)
CONSTANT_CAST_VECTOR_SPECIALIZATION(i4, float16)
CONSTANT_CAST_VECTOR_SPECIALIZATION(i4, bfloat16)
CONSTANT_CAST_VECTOR_SPECIALIZATION(i4, float)
CONSTANT_CAST_VECTOR_SPECIALIZATION(i4, double)

#undef CONSTANT_CAST_VECTOR_SPECIALIZATION

#define CONSTANT_WRITE_BUFFER_SPECIALIZATION(ET, SRC_TYPE) \
    template <>                                            \
    OPENVINO_API void Constant::write_lp_buffer<element::Type_t::ET>(const std::vector<SRC_TYPE>& source);

CONSTANT_WRITE_BUFFER_SPECIALIZATION(u1, bool)
CONSTANT_WRITE_BUFFER_SPECIALIZATION(u1, char)
CONSTANT_WRITE_BUFFER_SPECIALIZATION(u1, signed char)
CONSTANT_WRITE_BUFFER_SPECIALIZATION(u1, unsigned char)
CONSTANT_WRITE_BUFFER_SPECIALIZATION(u1, short)
CONSTANT_WRITE_BUFFER_SPECIALIZATION(u1, unsigned short)
CONSTANT_WRITE_BUFFER_SPECIALIZATION(u1, int)
CONSTANT_WRITE_BUFFER_SPECIALIZATION(u1, unsigned int)
CONSTANT_WRITE_BUFFER_SPECIALIZATION(u1, long)
CONSTANT_WRITE_BUFFER_SPECIALIZATION(u1, unsigned long)
CONSTANT_WRITE_BUFFER_SPECIALIZATION(u1, long long)
CONSTANT_WRITE_BUFFER_SPECIALIZATION(u1, unsigned long long)
CONSTANT_WRITE_BUFFER_SPECIALIZATION(u1, float8_e4m3)
CONSTANT_WRITE_BUFFER_SPECIALIZATION(u1, float8_e5m2)
CONSTANT_WRITE_BUFFER_SPECIALIZATION(u1, float16)
CONSTANT_WRITE_BUFFER_SPECIALIZATION(u1, bfloat16)
CONSTANT_WRITE_BUFFER_SPECIALIZATION(u1, float)
CONSTANT_WRITE_BUFFER_SPECIALIZATION(u1, double)

CONSTANT_WRITE_BUFFER_SPECIALIZATION(u4, bool)
CONSTANT_WRITE_BUFFER_SPECIALIZATION(u4, char)
CONSTANT_WRITE_BUFFER_SPECIALIZATION(u4, signed char)
CONSTANT_WRITE_BUFFER_SPECIALIZATION(u4, unsigned char)
CONSTANT_WRITE_BUFFER_SPECIALIZATION(u4, short)
CONSTANT_WRITE_BUFFER_SPECIALIZATION(u4, unsigned short)
CONSTANT_WRITE_BUFFER_SPECIALIZATION(u4, int)
CONSTANT_WRITE_BUFFER_SPECIALIZATION(u4, unsigned int)
CONSTANT_WRITE_BUFFER_SPECIALIZATION(u4, long)
CONSTANT_WRITE_BUFFER_SPECIALIZATION(u4, unsigned long)
CONSTANT_WRITE_BUFFER_SPECIALIZATION(u4, long long)
CONSTANT_WRITE_BUFFER_SPECIALIZATION(u4, unsigned long long)
CONSTANT_WRITE_BUFFER_SPECIALIZATION(u4, float8_e4m3)
CONSTANT_WRITE_BUFFER_SPECIALIZATION(u4, float8_e5m2)
CONSTANT_WRITE_BUFFER_SPECIALIZATION(u4, float16)
CONSTANT_WRITE_BUFFER_SPECIALIZATION(u4, bfloat16)
CONSTANT_WRITE_BUFFER_SPECIALIZATION(u4, float)
CONSTANT_WRITE_BUFFER_SPECIALIZATION(u4, double)

CONSTANT_WRITE_BUFFER_SPECIALIZATION(i4, bool)
CONSTANT_WRITE_BUFFER_SPECIALIZATION(i4, char)
CONSTANT_WRITE_BUFFER_SPECIALIZATION(i4, signed char)
CONSTANT_WRITE_BUFFER_SPECIALIZATION(i4, unsigned char)
CONSTANT_WRITE_BUFFER_SPECIALIZATION(i4, short)
CONSTANT_WRITE_BUFFER_SPECIALIZATION(i4, unsigned short)
CONSTANT_WRITE_BUFFER_SPECIALIZATION(i4, int)
CONSTANT_WRITE_BUFFER_SPECIALIZATION(i4, unsigned int)
CONSTANT_WRITE_BUFFER_SPECIALIZATION(i4, long)
CONSTANT_WRITE_BUFFER_SPECIALIZATION(i4, unsigned long)
CONSTANT_WRITE_BUFFER_SPECIALIZATION(i4, long long)
CONSTANT_WRITE_BUFFER_SPECIALIZATION(i4, unsigned long long)
CONSTANT_WRITE_BUFFER_SPECIALIZATION(i4, float8_e4m3)
CONSTANT_WRITE_BUFFER_SPECIALIZATION(i4, float8_e5m2)
CONSTANT_WRITE_BUFFER_SPECIALIZATION(i4, float16)
CONSTANT_WRITE_BUFFER_SPECIALIZATION(i4, bfloat16)
CONSTANT_WRITE_BUFFER_SPECIALIZATION(i4, float)
CONSTANT_WRITE_BUFFER_SPECIALIZATION(i4, double)

CONSTANT_WRITE_BUFFER_SPECIALIZATION(nf4, bool)
CONSTANT_WRITE_BUFFER_SPECIALIZATION(nf4, char)
CONSTANT_WRITE_BUFFER_SPECIALIZATION(nf4, signed char)
CONSTANT_WRITE_BUFFER_SPECIALIZATION(nf4, unsigned char)
CONSTANT_WRITE_BUFFER_SPECIALIZATION(nf4, short)
CONSTANT_WRITE_BUFFER_SPECIALIZATION(nf4, unsigned short)
CONSTANT_WRITE_BUFFER_SPECIALIZATION(nf4, int)
CONSTANT_WRITE_BUFFER_SPECIALIZATION(nf4, unsigned int)
CONSTANT_WRITE_BUFFER_SPECIALIZATION(nf4, long)
CONSTANT_WRITE_BUFFER_SPECIALIZATION(nf4, unsigned long)
CONSTANT_WRITE_BUFFER_SPECIALIZATION(nf4, long long)
CONSTANT_WRITE_BUFFER_SPECIALIZATION(nf4, unsigned long long)
CONSTANT_WRITE_BUFFER_SPECIALIZATION(nf4, float8_e4m3)
CONSTANT_WRITE_BUFFER_SPECIALIZATION(nf4, float8_e5m2)
CONSTANT_WRITE_BUFFER_SPECIALIZATION(nf4, float16)
CONSTANT_WRITE_BUFFER_SPECIALIZATION(nf4, bfloat16)
CONSTANT_WRITE_BUFFER_SPECIALIZATION(nf4, float)
CONSTANT_WRITE_BUFFER_SPECIALIZATION(nf4, double)

#undef CONSTANT_WRITE_BUFFER_SPECIALIZATION

}  // namespace v0
}  // namespace op
}  // namespace ov
