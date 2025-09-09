// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstring>
#include <variant>

#include "openvino/core/axis_set.hpp"
#include "openvino/core/axis_vector.hpp"
#include "openvino/core/coordinate_diff.hpp"
#include "openvino/core/rtti.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/element_type_traits.hpp"
#include "openvino/op/op.hpp"

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
        if constexpr (data::is_supported_type<T>) {
            data::fill(m_element_type, get_data_ptr_nc(), shape_size(m_shape), value);
        } else {
            proxy_type<T> proxy_value = value;
            data::fill(m_element_type, get_data_ptr_nc(), shape_size(m_shape), proxy_value);
        }
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

    /// \brief Construct a tensor constant from shared memory.
    ///
    /// The Constant can take ownership of shared memory if provided shared object is not null and manges memory
    /// lifetime.
    ///
    /// \param type   The element type of the tensor constant.
    /// \param shape  The shape of the tensor constant.
    /// \param data   The pointer to shared memory.
    /// \param so     The shared object to take it ownership.
    Constant(const element::Type& type, const Shape& shape, const void* data, std::shared_ptr<void> so);

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
    bool can_constant_fold(const OutputVector& inputs_values) const override;

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

    /// @brief Get constant buffer as vector of element type T.
    ///
    /// For low precision the vector do not perform bit unpacks.
    /// The returned vector has N elements where:
    /// - N is (elements count * (precision byte size / T byte size)) for standard precisions.
    /// - N is (byte size) for low precisions.
    ///
    /// @tparam T Output vector type which byte size must be less or equal of byte size of Constant's precision.
    /// @return Vector of N elements of Type T.
    template <typename T, typename std::enable_if<!std::is_same<bool, T>::value>::type* = nullptr>
    std::vector<T> get_vector() const {
        const auto p = get_data_ptr<T>();
        OPENVINO_ASSERT(p != nullptr, "Cannot create vector! Buffer is not allocated.");
        auto v = std::vector<T>(p, p + (get_byte_size() / sizeof(T)));
        if (!m_alloc_buffer_on_visit_attributes) {
            // result vector requires update when Constant share data (e.g read weight from IR binary file)
            set_unused_bits(v.data());
        }
        return v;
    }

    template <typename T, typename std::enable_if<std::is_same<bool, T>::value>::type* = nullptr>
    std::vector<T> get_vector() const {
        const auto p = get_data_ptr<T>();
        OPENVINO_ASSERT(p != nullptr, "Cannot create vector! Buffer is not allocated.");
        auto v = std::vector<T>(p, p + (get_byte_size() / sizeof(T)));
        return v;
    }

    /// \brief Return the Constant's value as a vector cast to type T
    ///
    /// \tparam T             Type to which data vector's entries will be cast.
    /// \param  num_elements  (Optional) Number of elements to cast. In default case returns all elements
    /// \return    Constant's data vector.
    template <typename T>
    std::vector<T> cast_vector(int64_t num_elements = -1) const {
        const auto num_elements_to_cast = get_num_elements_to_cast(num_elements);

        if constexpr (data::is_supported_type<T>) {
            std::vector<T> rc(num_elements_to_cast);

            if constexpr (std::is_same_v<bool, T>) {
                data::cast_n(m_element_type, get_data_ptr(), num_elements_to_cast, rc.begin());
            } else {
                data::cast_n(m_element_type, get_data_ptr(), num_elements_to_cast, rc.data());
            }
            return rc;
        } else {
            std::vector<proxy_type<T>> rc(num_elements_to_cast);
            data::cast_n(m_element_type, get_data_ptr(), num_elements_to_cast, rc.data());
            return {rc.begin(), rc.end()};
        }
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

    /// @brief Get view on constant data as tensor.
    /// @return ov::Tensor with constant data.
    const Tensor get_tensor_view() const;

    /// @return Constant's strides in bytes.
    const Strides& get_strides() const;

private:
    Constant(bool memset_allocation, const element::Type& type, const Shape& shape);

    size_t get_num_elements_to_cast(const int64_t n) const;

    /// \brief Sets buffer's not used bits to zero.
    ///
    /// In case of low precision there can be some storage area which is not used (not defined state).
    ///
    /// \param buffer  Pointer to buffer with Constant values.
    void set_unused_bits(void* buffer) const;

    void allocate_buffer(bool memset_allocation);

    void* get_data_ptr_nc();

    template <element::Type_t ET>
    typename ov::fundamental_type_for<ET>* get_data_ptr_nc() {
        OPENVINO_ASSERT(ET == get_element_type(), "get_data_ptr_nc() called for incorrect element type.");
        return static_cast<typename element_type_traits<ET>::value_type*>(get_data_ptr_nc());
    }

    template <typename T>
    void write_values(const std::vector<T>& source) {
        OPENVINO_ASSERT(source.size() == shape_size(m_shape), "Constant initializer does not match shape");

        if constexpr (!data::is_supported_type<T>) {
            std::vector<proxy_type<T>> temp{source.begin(), source.end()};
            data::copy_n(m_element_type, temp.data(), temp.size(), get_data_ptr_nc());
        } else if constexpr (std::is_same_v<bool, T>) {
            data::copy_n(m_element_type, source.begin(), source.size(), get_data_ptr_nc());
        } else {
            data::copy_n(m_element_type, source.data(), source.size(), get_data_ptr_nc());
        }
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

    template <class... Ts>
    struct Data {
        using value = std::variant<Ts...>;
        using pointer =
            std::variant<std::conditional_t<std::is_same_v<Ts, bool>, typename std::vector<Ts>::iterator, Ts*>...>;
        using const_pointer = std::variant<
            std::conditional_t<std::is_same_v<Ts, bool>, typename std::vector<Ts>::const_iterator, const Ts*>...>;

        template <class U>
        static constexpr auto is_supported_type = std::disjunction_v<std::is_same<U, Ts>...>;

        static void fill(const element::Type& type, void* dst, const size_t n, const value& value);
        static void copy_n(const element::Type& type, const_pointer src, const size_t n, void* dst);
        static void cast_n(const element::Type& type, const void* src, const size_t n, pointer dst);
    };

    template <class U>
    using proxy_type = std::conditional_t<std::is_integral_v<U>,
                                          long long,
                                          std::conditional_t<std::is_unsigned_v<U>, unsigned long long, double>>;

    using data = Data<bool,
                      char,
                      signed char,
                      unsigned char,
                      short,
                      unsigned short,
                      int,
                      unsigned int,
                      long,
                      unsigned long,
                      long long,
                      unsigned long long,
                      float,
                      double,
                      float4_e2m1,
                      float8_e4m3,
                      float8_e5m2,
                      float8_e8m0,
                      float16,
                      bfloat16,
                      std::string>;

    element::Type m_element_type{};
    Shape m_shape{};
    Strides m_byte_strides{};
    std::shared_ptr<ov::AlignedBuffer> m_data{};
    mutable std::atomic_bool m_all_elements_bitwise_identical{false};
    mutable std::atomic_bool m_all_elements_bitwise_identical_checked{false};
    bool m_alloc_buffer_on_visit_attributes{true};
};

template <>
OPENVINO_API void Constant::data::fill(const element::Type& type, void* dst, const size_t n, const value& value);

template <>
OPENVINO_API void Constant::data::copy_n(const element::Type& type, const_pointer src, const size_t n, void* dst);

template <>
OPENVINO_API void Constant::data::cast_n(const element::Type& type, const void* src, const size_t n, pointer dst);

}  // namespace v0
}  // namespace op
}  // namespace ov
