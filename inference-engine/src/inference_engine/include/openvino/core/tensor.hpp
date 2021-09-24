// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for the OpenVINO Runtime tensor API
 *
 * @file openvino/core/tensor.hpp
 */
#pragma once

#include "ie_api.h"
#include "openvino/core/allocator.hpp"
#include "openvino/core/coordinate.hpp"
#include "openvino/core/core_visibility.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/common.hpp"

namespace InferenceEngine {
class Blob;
}  // namespace InferenceEngine

namespace ov {
namespace runtime {
class InferRequest;
class RemoteContext;
class VariableState;
}  // namespace runtime

/**
 * @brief Memory access and interpretation API
 *
 * It can throw exceptions safely for the application, where it is properly handled.
 */
class INFERENCE_ENGINE_API_CLASS(Tensor) {
protected:
    std::shared_ptr<void> _so;        //!< Refernce to dynamicly loaded library
    std::shared_ptr<ie::Blob> _impl;  //!< Shared pointer to internal tensor representation

    /**
     * @brief Constructs Tensor from the initialized std::shared_ptr
     * @param so Plugin to use. This is required to ensure that Tensor can work properly even if plugin object is
     * destroyed.
     * @param impl Initialized shared pointer
     */
    Tensor(const std::shared_ptr<void>& so, const std::shared_ptr<ie::Blob>& impl);

    friend class ov::runtime::InferRequest;
    friend class ov::runtime::RemoteContext;
    friend class ov::runtime::VariableState;

public:
    /**
     * @brief Default constructor
     */
    Tensor() = default;

    /**
     * @brief Constructs Tensor using element type and shape. Allocate internal host storage using default allocator
     * @param type Tensor element type
     * @param shape Tensor shape
     * @param allocator allocates memory for internal tensor storage
     */
    Tensor(const element::Type type, const Shape& shape, const Allocator& allocator = {});

    /**
     * @brief Constructs Tensor using element type and shape. Wraps allocated host memory.
     * @note Dose not own the memory
     * @param type Tensor element type
     * @param shape Tensor shape
     * @param ptr Pointer to allocate memory
     * @param size Optional size of allocated host memory in elements. If it is not used the size of memory supposed to
     * be not less then ov::shape_size(shape) * type.size() in bytes.
     * @param strides Optional element strides paramters. It is supposed that strides is equal to shape if it is not
     * used
     */
    Tensor(const element::Type type, const Shape& shape, void* ptr, const size_t size = 0, const Strides& strides = {});

    /**
     * @brief constructs region of interest (ROI) tensor form other tensor.
     * @note Dose not own the memory
     * @param other original tensor
     * @param begin start coordinate of ROI object inside of the original object.
     * @param end end coordinate of ROI object inside of the original object.
     */
    Tensor(const Tensor& other, const Coordinate& begin, const Coordinate& end);

    /**
     * @brief Set new shape for tensor, deallocate/allocate if new total size is bigger than previous one.
     *
     * @param shape new shape
     */
    void set_shape(const ov::Shape& shape);

    /**
     * @return tensor element type
     */
    element::Type get_element_type() const;

    /**
     * @return tensor element type
     */
    Shape get_shape() const;

    /**
     * @brief By default, returns the total number of elements (a product of all the dims or 1 for scalar)
     *
     * Return value and its interpretation heavily depend on the internal tensor implementation
     *
     * @return The total number of elements
     */
    size_t get_size() const;

    /**
     * @brief Returns the size of the current Tensor in bytes.
     * @return Tensor's size in bytes
     */
    size_t get_byte_size() const;

    /**
     * @return Tensor's strides
     */
    Strides get_strides() const;

    /**
     * @return host pointer to tensor memory
     * @param type Optional type parameter.
     * If type parameter is specified throws exception
     * if specified type fundamental type dose not match with tensor element type fundamental type
     */
    void* data(const element::Type type = {}) const;

    /**
     * @return host pointer to tensor memory casted to specified type.
     * Throws exception if specified type dose not match with tensor element type
     */
    template <typename T>
    T* data() const {
        return static_cast<T*>(data(element::from<T>()));
    }

    /**
     * @brief Checks if current Tensor object is not initialized
     * @return true if current Tensor object is not initialized, false - otherwise
     */
    bool operator!() const noexcept;

    /**
     * @brief Checks if current Tensor object is initialized
     * @return true if current Tensor object is initialized, false - otherwise
     */
    explicit operator bool() const noexcept;
};
}  // namespace ov