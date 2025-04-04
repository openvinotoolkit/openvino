// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for the OpenVINO Runtime tensor API
 *
 * @file openvino/runtime/tensor.hpp
 */
#pragma once

#include <filesystem>
#include <type_traits>

#include "openvino/core/coordinate.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/rtti.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/allocator.hpp"

namespace ov {

class Tensor;
class ITensor;

namespace util {
ov::Tensor make_tensor(const std::shared_ptr<ov::ITensor>& tensor, const std::shared_ptr<void>& so);
void get_tensor_impl(const ov::Tensor& tensor, std::shared_ptr<ov::ITensor>& tensor_impl, std::shared_ptr<void>& so);
}  // namespace util

namespace op {
namespace util {
class VariableValue;
}  // namespace util
}  // namespace op

/**
 * @brief Tensor API holding host memory
 * It can throw exceptions safely for the application, where it is properly handled.
 * @ingroup ov_runtime_cpp_api
 */
class OPENVINO_API Tensor {
protected:
    std::shared_ptr<ITensor> _impl;  //!< Shared pointer to internal tensor representation
    std::shared_ptr<void> _so;       //!< Reference to dynamically loaded library

    /**
     * @brief Constructs Tensor from the initialized std::shared_ptr
     * @param impl Initialized shared pointer
     * @param so Plugin to use. This is required to ensure that Tensor can work properly even if plugin object is
     * destroyed.
     */
    Tensor(const std::shared_ptr<ITensor>& impl, const std::shared_ptr<void>& so);

    friend class ov::op::util::VariableValue;
    friend ov::Tensor ov::util::make_tensor(const std::shared_ptr<ov::ITensor>& tensor,
                                            const std::shared_ptr<void>& so);
    friend void ov::util::get_tensor_impl(const ov::Tensor& tensor,
                                          std::shared_ptr<ov::ITensor>& tensor_impl,
                                          std::shared_ptr<void>& so);

public:
    /// @brief Default constructor
    Tensor() = default;

    /**
     * @brief Copy constructor with adding new shared object
     *
     * @param other Original tensor
     * @param so Shared object
     */
    Tensor(const Tensor& other, const std::shared_ptr<void>& so);

    /// @brief Default copy constructor
    /// @param other other Tensor object
    Tensor(const Tensor& other) = default;

    /// @brief Default copy assignment operator
    /// @param other other Tensor object
    /// @return reference to the current object
    Tensor& operator=(const Tensor& other) = default;

    /// @brief Default move constructor
    /// @param other other Tensor object
    Tensor(Tensor&& other) = default;

    /// @brief Default move assignment operator
    /// @param other other Tensor object
    /// @return reference to the current object
    Tensor& operator=(Tensor&& other) = default;

    /**
     * @brief Destructor preserves unloading order of implementation object and reference to library
     */
    ~Tensor();

    /**
     * @brief Checks openvino tensor type
     * @param tensor a tensor which type will be checked
     * @throw Exception if type check with specified tensor is not pass
     */
    static void type_check(const Tensor& tensor);

    /**
     * @brief Constructs Tensor using element type and shape. Allocate internal host storage using default allocator
     * @param type Tensor element type
     * @param shape Tensor shape
     * @param allocator allocates memory for internal tensor storage
     */
    Tensor(const element::Type& type, const Shape& shape, const Allocator& allocator = {});

    /**
     * @brief Constructs Tensor using element type and shape. Wraps allocated host memory.
     * @note Does not perform memory allocation internally
     * @param type Tensor element type
     * @param shape Tensor shape
     * @param host_ptr Pointer to pre-allocated host memory with initialized objects
     * @param strides Optional strides parameters in bytes. Strides are supposed to be computed automatically based
     * on shape and element size
     */
    Tensor(const element::Type& type, const Shape& shape, void* host_ptr, const Strides& strides = {});

    /**
     * @brief Constructs Tensor using element type and shape. Wraps allocated host memory as read only.
     * @note Does not perform memory allocation internally
     * @param type Tensor element type
     * @param shape Tensor shape
     * @param host_ptr Pointer to pre-allocated host memory with initialized objects
     * @param strides Optional strides parameters in bytes. Strides are supposed to be computed automatically based
     * on shape and element size
     */
    Tensor(const element::Type& type, const Shape& shape, const void* host_ptr, const Strides& strides = {});

    /**
     * @brief Constructs Tensor using port from node. Allocate internal host storage using default allocator
     * @param port port from node
     * @param allocator allocates memory for internal tensor storage
     */
    Tensor(const ov::Output<const ov::Node>& port, const Allocator& allocator = {});

    /**
     * @brief Constructs Tensor using port from node. Wraps allocated host memory.
     * @note Does not perform memory allocation internally
     * @param port port from node
     * @param host_ptr Pointer to pre-allocated host memory with initialized objects
     * @param strides Optional strides parameters in bytes. Strides are supposed to be computed automatically based
     * on shape and element size
     */
    Tensor(const ov::Output<const ov::Node>& port, void* host_ptr, const Strides& strides = {});

    /**
     * @brief Constructs Tensor using port from node. Wraps allocated host memory as read only.
     * @note Does not perform memory allocation internally
     * @param port port from node
     * @param host_ptr Pointer to pre-allocated host memory with initialized objects
     * @param strides Optional strides parameters in bytes. Strides are supposed to be computed automatically based
     * on shape and element size
     */
    Tensor(const ov::Output<const ov::Node>& port, const void* host_ptr, const Strides& strides = {});

    /**
     * @brief Constructs region of interest (ROI) tensor form another tensor.
     * @note Does not perform memory allocation internally
     * @param other original tensor
     * @param begin start coordinate of ROI object inside of the original object.
     * @param end end coordinate of ROI object inside of the original object.
     * @note A Number of dimensions in `begin` and `end` must match number of dimensions in `other.get_shape()`
     */
    Tensor(const Tensor& other, const Coordinate& begin, const Coordinate& end);

    /**
     * @brief Set new shape for tensor, deallocate/allocate if new total size is bigger than previous one.
     * @note Memory allocation may happen
     * @param shape A new shape
     */
    void set_shape(const ov::Shape& shape);

    /**
     * @return A tensor element type
     */
    const element::Type& get_element_type() const;

    /**
     * @return A tensor shape
     */
    const Shape& get_shape() const;

    /**
     * @brief Copy tensor, destination tensor should have the same element type and shape
     *
     * @param dst destination tensor
     */
    void copy_to(ov::Tensor dst) const;

    /**
     * @brief Reports whether the tensor is continuous or not
     *
     * @return true if tensor is continuous
     */
    bool is_continuous() const;

    /**
     * @brief Returns the total number of elements (a product of all the dims or 1 for scalar)
     * @return The total number of elements
     */
    size_t get_size() const;

    /**
     * @brief Returns the size of the current Tensor in bytes.
     * @return Tensor's size in bytes
     */
    size_t get_byte_size() const;

    /**
     * @return Tensor's strides in bytes
     */
    Strides get_strides() const;

    /**
     * @brief Provides an access to the underlying host memory
     * @param type Optional type parameter.
     * @note If type parameter is specified, the method throws an exception
     * if specified type's fundamental type does not match with tensor element type's fundamental type
     * @return A host pointer to tensor memory
     * @{
     */
#ifndef IN_OV_COMPONENT
    OPENVINO_DEPRECATED("This function will return const void* in 2026.0. Check if used correctly")
#endif
    void* data(const element::Type& type = {}) const;
    void* data(const element::Type& type = {});
    /// @}

    /**
     * @brief Provides an access to the underlying host memory casted to type `T`
     * @return A host pointer to tensor memory casted to specified type `T`.
     * @note Throws exception if specified type does not match with tensor element type
     * @{
     */
    template <typename T, typename datatype = std::decay_t<T>>
#ifndef IN_OV_COMPONENT
    OPENVINO_DEPRECATED("This function will return const T* in 2026.0. Check if used correctly")
#endif
    T* data() const {
        OPENVINO_SUPPRESS_DEPRECATED_START  // keep until 2026.0 release
            return static_cast<T*>(data(element::from<datatype>()));
        OPENVINO_SUPPRESS_DEPRECATED_END  // keep until 2026.0 release
    }

    template <typename T, typename datatype = std::decay_t<T>>
    T* data() {
        if constexpr (std::is_const_v<T>) {
            OPENVINO_SUPPRESS_DEPRECATED_START  // keep until 2026.0 release
                return std::as_const(*this)
                    .data<T>();
            OPENVINO_SUPPRESS_DEPRECATED_END  // keep until 2026.0 release
        } else {
            return static_cast<T*>(data(element::from<datatype>()));
        }
    }
    /// @}

    /**
     * @brief Checks if current Tensor object is not initialized
     * @return `true` if current Tensor object is not initialized, `false` - otherwise
     */
    bool operator!() const noexcept;

    /**
     * @brief Checks if current Tensor object is initialized
     * @return `true` if current Tensor object is initialized, `false` - otherwise
     */
    explicit operator bool() const noexcept;

    /**
     * @brief Checks if the Tensor object can be cast to the type T
     *
     * @tparam T Type to be checked. Must represent a class derived from the Tensor
     * @return true if this object can be dynamically cast to the type const T*. Otherwise, false
     */
    template <typename T>
    std::enable_if_t<std::is_base_of_v<Tensor, T>, bool> is() const noexcept {
        try {
            T::type_check(*this);
        } catch (...) {
            return false;
        }
        return true;
    }

    /**
     * @brief Casts this Tensor object to the type T.
     *
     * @tparam T Type to cast to. Must represent a class derived from the Tensor
     * @return T object
     */
    template <typename T>
    const std::enable_if_t<std::is_base_of_v<Tensor, T>, T> as() const {
        T::type_check(*this);
        return *static_cast<const T*>(this);
    }
};

/**
 * @brief A vector of Tensor's
 */
using TensorVector = std::vector<Tensor>;

/// \brief Read a tensor content from a file. Only raw data is loaded.
/// \param file_name Path to file to read.
/// \param element_type Element type, when not specified the it is assumed as element::u8.
/// \param shape Shape for resulting tensor. If provided shape is static, specified number of elements is read only.
///              File should contain enough bytes, an exception is raised otherwise.
///              One of the dimensions can be dynamic. In this case it will be determined automatically based on the
///              length of the file content and `offset`. Default value is [?].
/// \param offset_in_bytes Read file starting from specified offset. Default is 0. The remining size of the file should
/// be compatible with shape.
/// \param mmap Use mmap that postpones real read from file until data is accessed. If mmap is used, the file
///             should not be modified until returned tensor is destroyed.
OPENVINO_API
Tensor read_tensor_data(const std::filesystem::path& file_name,
                        const element::Type& element_type = element::u8,
                        const PartialShape& shape = PartialShape::dynamic(1),
                        std::size_t offset_in_bytes = 0,
                        bool mmap = true);
}  // namespace ov
