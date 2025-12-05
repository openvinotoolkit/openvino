// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <mutex>

#include "executable.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov {
namespace runtime {

/// \brief Interface to a generic backend.
///
/// Backends are responsible for function execution and value allocation.
class Backend {
public:
    virtual ~Backend();
    /// \brief Create a new Backend object
    /// \param type The name of a registered backend, such as "CPU" or "GPU".
    ///   To select a subdevice use "GPU:N" where s`N` is the subdevice number.
    /// \param must_support_dynamic If `true`, the returned `Backend` object
    ///    will support dynamic tensors. If the underlying backend has native
    ///    support for dynamic tensors, then that backend object will be
    ///    returned directly. Otherwise, it will be wrapped with
    ///    DynamicWrapperBackend. This feature is EXPERIMENTAL.
    /// \returns shared_ptr to a new Backend or nullptr if the named backend
    ///   does not exist.
    static std::shared_ptr<Backend> create();

    /// \brief Create a tensor specific to this backend
    /// This call is used when an output is dynamic and not known until execution time. When
    /// passed as an output to a function the tensor will have a type and shape after executing
    /// a call.
    /// \returns shared_ptr to a new backend-specific tensor
    virtual ov::Tensor create_tensor() = 0;

    /// \brief Create a tensor specific to this backend
    /// \param element_type The type of the tensor element
    /// \param shape The shape of the tensor
    /// \returns shared_ptr to a new backend-specific tensor
    virtual ov::Tensor create_tensor(const ov::element::Type& element_type, const Shape& shape) = 0;

    /// \brief Create a tensor specific to this backend
    /// \param element_type The type of the tensor element
    /// \param shape The shape of the tensor
    /// \param memory_pointer A pointer to a buffer used for this tensor. The size of the buffer
    ///     must be sufficient to contain the tensor. The lifetime of the buffer is the
    ///     responsibility of the caller.
    /// \returns shared_ptr to a new backend-specific tensor
    virtual ov::Tensor create_tensor(const ov::element::Type& element_type,
                                     const Shape& shape,
                                     void* memory_pointer) = 0;

    /// \brief Create a tensor of C type T specific to this backend
    /// \param shape The shape of the tensor
    /// \returns shared_ptr to a new backend specific tensor
    template <typename T>
    ov::Tensor create_tensor(const Shape& shape) {
        return create_tensor(element::from<T>(), shape);
    }

    /// \brief Compiles a Function.
    /// \param func The function to compile
    /// \returns compiled function or nullptr on failure
    virtual std::shared_ptr<Executable> compile(std::shared_ptr<ov::Model> model) = 0;
};

}  // namespace runtime
}  // namespace ov
