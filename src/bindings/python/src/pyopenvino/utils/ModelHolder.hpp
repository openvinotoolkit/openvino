// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pybind11/pybind11.h>

namespace ov {

class Model;
class CompiledModel;

}  // namespace ov

template <typename T>
class ModelHolder {
    static_assert(std::is_same<T, ov::Model>::value, "ModelHolder can only hold ov::Model");
    std::shared_ptr<ov::Model> m_model;
    std::shared_ptr<ov::CompiledModel> m_compiled_model;

public:
    ModelHolder() = default;
    ModelHolder(const ModelHolder&) = default;
    ModelHolder(ModelHolder&&) = default;
    ModelHolder& operator=(const ModelHolder&) = default;
    ModelHolder& operator=(ModelHolder&&) = default;

    // construct from std::shared_ptr<ov::Model>
    ModelHolder(const std::shared_ptr<ov::Model>& model) : m_model(model) {}
    ModelHolder(std::shared_ptr<ov::Model>&& model) : m_model(std::move(model)) {}

    // special constructor for CompiledModel::get_runtime_model()
    ModelHolder(std::shared_ptr<ov::Model>&& model, std::shared_ptr<ov::CompiledModel>& compiled_model)
        : m_model(std::move(model)),
          m_compiled_model(compiled_model) {}

    // calls shared_from_this() automatically by the constructor of std::shared_ptr
    ModelHolder(ov::Model* model) : m_model(model) {}

    // make sure the compiled model is destructed after the runtime model to
    // keep the dynamic-loaded library alive, as described in issue #18388
    ~ModelHolder() noexcept {
        m_model.reset();
    }

    // required by pybind11
    ov::Model* get() const noexcept {
        return m_model.get();
    }
};

PYBIND11_DECLARE_HOLDER_TYPE(T, ModelHolder<T>);
