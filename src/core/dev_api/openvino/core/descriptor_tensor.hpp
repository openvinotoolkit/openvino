// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <unordered_set>

#include "openvino/core/partial_shape.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov {
namespace descriptor {

/// @brief Defines tensor name port separator.
inline constexpr auto port_separator = ':';
/// @brief Defines unique name separator.
inline constexpr auto unique_name_sep = '_';

class Tensor;
class Input;
class Output;

// To change Tensor element type please change the Parameter type.
OPENVINO_API
void set_element_type(Tensor& tensor, const element::Type& elemenet_type);

// To change Tensor type please change the Parameter type.
OPENVINO_API
void set_tensor_type(Tensor& tensor, const element::Type& element_type, const PartialShape& pshape);

/**
 * @brief Set destination tensor names as copy of all names from source tensor all tensor names.
 *
 * @param dst  The tensor descriptor to set names.
 * @param src  The tensor descriptor as from which names will be copied.
 */
OPENVINO_API
void copy_tensor_names(Tensor& dst, const Tensor& src);

/**
 * @brief Add names to destination tensor by copying of all names from source tensor all tensor names.
 *
 * If source tensor is parameter's tensor names are not copied.
 *
 * @param dst  The tensor descriptor to set names.
 * @param src  The tensor descriptor as from which names will be copied.
 */
OPENVINO_API
void add_not_parameter_names(Tensor& dst, const Tensor& src);

/** @brief Tensor descriptor interface. */
class OPENVINO_API ITensorDescriptor {
public:
    virtual const element::Type& get_element_type() const = 0;
    virtual const PartialShape& get_partial_shape() const = 0;
    virtual const Shape& get_shape() const = 0;
    virtual void set_type_shape(const element::Type& et, const PartialShape& shape) = 0;

    virtual void set_names(const std::unordered_set<std::string>& names) = 0;
    virtual void add_names(const std::unordered_set<std::string>& names) = 0;
    virtual const std::unordered_set<std::string>& get_names() const = 0;
    virtual const std::unordered_set<std::string>& get_all_names() const = 0;
    virtual const std::string& get_any_name() const = 0;

    virtual RTMap& rt_map() = 0;
    virtual const RTMap& rt_map() const = 0;
    virtual size_t pointer_hash() const noexcept = 0;

protected:
    virtual ~ITensorDescriptor();
};

/** @brief The TensorExtension defines developer API for ov::descriptor::Tensor. */
struct OPENVINO_API TensorExtension {
    /**
     * @brief Get the tensor descriptor object
     *
     * @param tensor Tensor descriptor to access its implementation.
     * @return Reference to Tensor description implementation.
     */
    static const ITensorDescriptor& get_descriptor(const Tensor& tensor);
    static std::shared_ptr<ITensorDescriptor>& get_descriptor_ptr(Tensor& tensor);

    /**
     * @brief The hasher of shared pointer Tensor descriptor.
     */
    struct OPENVINO_API Hasher {
        size_t operator()(const std::shared_ptr<Tensor>& tensor) const;
    };

    /**
     * @brief The comparator of shared pointer Tensor descriptor.
     */
    struct OPENVINO_API Equal {
        bool operator()(const std::shared_ptr<Tensor>& lhs, const std::shared_ptr<Tensor>& rhs) const;
    };
};

/**
 * @brief Set input descriptor as shared tensor on output descriptor.
 *
 * @param output_descriptor  Descriptor to set shared tensor.
 * @param input_descriptor   Input descriptor to set in output as shared tensor.
 * @param is_parameter       Flag to set shared tensor as parameter tensor.
 */
OPENVINO_API void set_shared_tensor(Output& output_descriptor, const Input& input_descriptor, bool is_parameter);

/**
 * @brief Retrieves the set of output names assigned to tensor descriptor.
 *
 * This function returns tensor descriptor names:
 * - same as ov::descriptor::Tensor::get_names() for regular descriptor.
 * - return specific output names for shared tensor.
 *
 * @param descriptor The tensor descriptor to get names.
 * @return The set of output names.
 */
OPENVINO_API const std::unordered_set<std::string>& get_assigned_names(const Tensor& descriptor);

}  // namespace descriptor
}  // namespace ov
