//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include <memory>
#include <mutex>

#include "backend_visibility.hpp"
#include "executable.hpp"
#include "ngraph/function.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"
#include "ngraph/util.hpp"
#include "performance_counter.hpp"

namespace ngraph
{
    namespace runtime
    {
        class Tensor;
        class Backend;
    }
}

/// \brief Interface to a generic backend.
///
/// Backends are responsible for function execution and value allocation.
class BACKEND_API ngraph::runtime::Backend
{
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
    static std::shared_ptr<Backend> create(const std::string& type,
                                           bool must_support_dynamic = false);

    /// \brief Query the list of registered devices
    /// \returns A vector of all registered devices.
    static std::vector<std::string> get_registered_devices();

    /// \brief Create a tensor specific to this backend
    /// This call is used when an output is dynamic and not known until execution time. When
    /// passed as an output to a function the tensor will have a type and shape after executing
    /// a call.
    /// \returns shared_ptr to a new backend-specific tensor
    virtual std::shared_ptr<ngraph::runtime::Tensor> create_tensor() = 0;

    /// \brief Create a tensor specific to this backend
    /// \param element_type The type of the tensor element
    /// \param shape The shape of the tensor
    /// \returns shared_ptr to a new backend-specific tensor
    virtual std::shared_ptr<ngraph::runtime::Tensor>
        create_tensor(const ngraph::element::Type& element_type, const Shape& shape) = 0;

    /// \brief Create a tensor specific to this backend
    /// \param element_type The type of the tensor element
    /// \param shape The shape of the tensor
    /// \param memory_pointer A pointer to a buffer used for this tensor. The size of the buffer
    ///     must be sufficient to contain the tensor. The lifetime of the buffer is the
    ///     responsibility of the caller.
    /// \returns shared_ptr to a new backend-specific tensor
    virtual std::shared_ptr<ngraph::runtime::Tensor> create_tensor(
        const ngraph::element::Type& element_type, const Shape& shape, void* memory_pointer) = 0;

    /// \brief Create a tensor of C type T specific to this backend
    /// \param shape The shape of the tensor
    /// \returns shared_ptr to a new backend specific tensor
    template <typename T>
    std::shared_ptr<ngraph::runtime::Tensor> create_tensor(const Shape& shape)
    {
        return create_tensor(element::from<T>(), shape);
    }

    /// \brief Create a dynamic tensor specific to this backend, if the backend supports dynamic
    ///        tensors.
    /// \param element_type The type of the tensor element
    /// \param shape The shape of the tensor
    /// \returns shared_ptr to a new backend-specific tensor
    /// \throws std::invalid_argument if the backend does not support dynamic tensors
    virtual std::shared_ptr<ngraph::runtime::Tensor>
        create_dynamic_tensor(const ngraph::element::Type& element_type, const PartialShape& shape);

    /// \returns `true` if this backend supports dynamic tensors, else `false`.
    virtual bool supports_dynamic_tensors() { return false; }
    /// \brief Compiles a Function.
    /// \param func The function to compile
    /// \returns compiled function or nullptr on failure
    virtual std::shared_ptr<Executable> compile(std::shared_ptr<Function> func,
                                                bool enable_performance_data = false) = 0;

    /// \brief Loads a previously saved Executable object from a stream.
    /// \param input_stream the opened input stream containing the saved Executable
    /// \returns A compiled function or throws an exception on error
    virtual std::shared_ptr<Executable> load(std::istream& input_stream);

    /// \brief Test if a backend is capable of supporting an op
    /// \param node is the op to test.
    /// \returns true if the op is supported, false otherwise.
    virtual bool is_supported(const Node& node) const;

    /// \brief Allows sending backend specific configuration. The map contains key, value pairs
    ///     specific to a particluar backend. The definition of these key, value pairs is
    ///     defined by each backend.
    /// \param config The configuration map sent to the backend
    /// \param error An error string describing any error encountered
    /// \returns true if the configuration is supported, false otherwise. On false the error
    ///     parameter value is valid.
    virtual bool set_config(const std::map<std::string, std::string>& config, std::string& error);

    static void set_backend_shared_library_search_directory(const std::string& path);
    static const std::string& get_backend_shared_library_search_directory();

    /// \brief Get the version of the backend
    /// The default value of 0.0.0 is chosen to be a parsable version number
    virtual std::string get_version() const { return "0.0.0"; }

private:
    // mutex to modify s_backend_shared_library_search_directory thread safe
    static std::mutex m_mtx;
    static std::string s_backend_shared_library_search_directory;
};
