// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "backend.hpp"
#include "cache.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/tensor.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace dynamic
        {
            class DynamicBackend;
            class DynamicExecutable;
            class DynamicTensor;
        }
    }
}

///
/// \brief Wrapper class used to provide dynamic tensor support on backends
///        that otherwise do not support dynamic tensors.
///
/// The main function of this class is to intercept `create_dynamic_tensor`
/// and `compile`:
///
/// * `create_dynamic_tensor` will return a special `DynamicTensor` object
///   whose shape can be updated after creation. Internally, `DynamicTensor`
///   wraps static tensors managed by the wrapped backend.
/// * `compile` will return a special `DynamicExecutable` object, which allows
///   dynamic shapes to be supported via graph cloning.
///
/// This class is instantiated by `ngraph::runtime::Backend::create`.
///
class ngraph::runtime::dynamic::DynamicBackend : public Backend
{
public:
    DynamicBackend(std::shared_ptr<ngraph::runtime::Backend> wrapped_backend);

    std::shared_ptr<Tensor> create_tensor() override;

    std::shared_ptr<Tensor>
        create_tensor(const element::Type& type, const Shape& shape, void* memory_pointer) override;

    std::shared_ptr<Tensor> create_tensor(const element::Type& type, const Shape& shape) override;

    std::shared_ptr<Tensor> create_dynamic_tensor(const element::Type& type,
                                                  const PartialShape& shape) override;

    bool supports_dynamic_tensors() override { return true; }
    std::shared_ptr<Executable> compile(std::shared_ptr<Function> function,
                                        bool enable_performance_data = false) override;

private:
    std::shared_ptr<ngraph::runtime::Backend> m_wrapped_backend;
};

///
/// \brief Wrapper class used to provide an Executable that supports dynamic
///        tensors on top of a backend that does not support dynamic tensors
///        natively.
///
/// This class intercepts `call` and:
///
/// 1. creates a clone of the stored function with shapes tailored to the
///    actual runtime inputs;
/// 2. compiles the clone using the wrapped backend;
/// 3. fowards the input tensors to the clone executable for actual execution.
///
/// `DynamicExecutable` objects are produced by `DynamicBackend::compile()`.
///
class ngraph::runtime::dynamic::DynamicExecutable : public ngraph::runtime::Executable
{
public:
    DynamicExecutable(std::shared_ptr<Function> wrapped_function,
                      std::shared_ptr<ngraph::runtime::Backend> wrapped_backend,
                      bool enable_performance_collection = false);
    bool call(const std::vector<std::shared_ptr<runtime::Tensor>>& outputs,
              const std::vector<std::shared_ptr<runtime::Tensor>>& inputs) override;

private:
    std::shared_ptr<ngraph::Function> m_wrapped_function;
    std::shared_ptr<ngraph::runtime::Backend> m_wrapped_backend;
    std::shared_ptr<ngraph::runtime::LRUCache> m_lru =
        std::make_shared<ngraph::runtime::LRUCache>();
    bool m_enable_performance_collection;
};

///
/// \brief Wrapper class used to emulate dynamic tensors on top of a backend
///        that does not support dynamic tensors natively.
///
/// The behavior of a dynamic tensor extends that of `runtime::Tensor` as
/// follows:
///
/// 1. `get_partial_shape()` returns a `PartialShape` representing all shapes
///    this tensor could possibly take on at execution time.
/// 2. `get_shape()` returns a `Shape` representing the _exact_ shape of this
///    tensor's current value. (If the tensor has not yet been assigned a
///    value, `get_shape()` throws an exception.)
/// 3. `make_storage()` allocates storage for a value of a specific element
///    type and shape, which must be consistent with the partial shape/element
///    type. Once storage has been allocated, `get_shape()` can safely be
///    called until the storage has been released via `release_storage()`.
/// 4. `release_storage()` unassigns previously assigned storage.
///
class ngraph::runtime::dynamic::DynamicTensor : public ngraph::runtime::Tensor
{
public:
    DynamicTensor(const element::Type& element_type,
                  const PartialShape& shape,
                  const std::shared_ptr<runtime::Backend>& wrapped_backend);
    virtual size_t get_size_in_bytes() const override;
    virtual size_t get_element_count() const override;
    virtual const element::Type& get_element_type() const override;
    virtual const ngraph::Shape& get_shape() const override;
    virtual void write(const void* p, size_t n) override;
    virtual void read(void* p, size_t n) const override;
    bool has_storage() const;
    void release_storage();
    void make_storage(const element::Type& element_type, const PartialShape& shape);
    const std::shared_ptr<ngraph::runtime::Tensor>& get_wrapped_tensor() const;

private:
    std::shared_ptr<ngraph::runtime::Tensor> m_wrapped_tensor;
    std::shared_ptr<ngraph::runtime::Backend> m_wrapped_backend;
};
