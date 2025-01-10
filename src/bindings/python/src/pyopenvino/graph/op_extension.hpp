// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pybind11/pybind11.h>

#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "openvino/core/op_extension.hpp"
#include "pyopenvino/graph/op.hpp"

namespace py = pybind11;

class PyOpExtension : public ov::BaseOpExtension {
public:
    PyOpExtension(const py::object& dtype) : py_handle_dtype{dtype} {
        py::object py_issubclass = py::module::import("builtins").attr("issubclass");
        if (!py_issubclass(dtype, py::type::of<PyOp>()).cast<bool>()) {
            std::stringstream str;
            str << "Unsupported data type : '" << dtype << "' is passed as an argument.";
            OPENVINO_THROW(str.str());
        }

        py::object type_info;
        try {
            // get_type_info() is a static method
            std::cout << "before static methods" << std::endl;
            type_info = py_handle_dtype.attr("get_type_info")();
            std::cout << "after static methods" << std::endl;
        } catch (const std::exception&) {
            try {
                //  get_type_info() is a class method
                std::cout << "before class methods" << std::endl;
                auto obj = py_handle_dtype();
                std::cout << "afte obj" << std::endl;
                type_info = obj.attr("get_type_info")();
                std::cout << "afte class methods" << std::endl;
            } catch (const std::exception &exc) {
                OPENVINO_THROW("Creation of OpExtension failed: ", exc.what());
            }
        }
        if (!py::isinstance<ov::DiscreteTypeInfo>(type_info)) {
            OPENVINO_THROW("operation type_info must be an instance of DiscreteTypeInfo, but ", py::str(py::type::of(type_info)), " is passed.");
        }
        m_type_info = type_info.cast<std::shared_ptr<ov::DiscreteTypeInfo>>();
        OPENVINO_ASSERT(m_type_info->name != nullptr && m_type_info->version_id != nullptr,
                        "Extension type should have information about operation set and operation type.");
    }

    const ov::DiscreteTypeInfo& get_type_info() const override;

    ov::OutputVector create(const ov::OutputVector& inputs, ov::AttributeVisitor& visitor) const override;

    std::vector<ov::Extension::Ptr> get_attached_extensions() const override;

private:
    py::object py_handle_dtype;  // Holds the Python object to manage its lifetime
    std::shared_ptr<ov::DiscreteTypeInfo> m_type_info; 
};

void regclass_graph_OpExtension(py::module m);
