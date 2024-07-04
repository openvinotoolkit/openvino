// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pybind11/pybind11.h>

#include "openvino/core/model.hpp"  // ov::Model

namespace py = pybind11;

class ModelWrapper {
public:
    ModelWrapper(std::shared_ptr<ov::Model> model) : m_model(new std::shared_ptr<ov::Model>(std::move(model))) {} 

    // TODO: rename me???
    void erase() {
        std::cout << "WTF, use count is " << m_model->use_count() << std::endl;
        m_model->reset();
        std::cout << "WTF2, use count is " << m_model->use_count() << std::endl;
        delete m_model;
        std::cout << "I deleted the model!!!" << std::endl;
        m_model = nullptr;
    }

    const ov::Model& get_model() const {
        return **m_model;
    }

    ov::Model& get_model() {
        return **m_model;
    }

private:
    // Original ov::Model class that is held by this wrapper
    std::shared_ptr<ov::Model>* m_model;
};

void regclass_graph_Model(py::module m);
