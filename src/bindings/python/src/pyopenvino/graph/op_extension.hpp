// Copyright (C) 2018-2025 Intel Corporation
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
            type_info = py_handle_dtype().attr("get_type_info")();
        } catch (const std::exception &exc) {
            OPENVINO_THROW("Creation of OpExtension failed: ", exc.what());
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
