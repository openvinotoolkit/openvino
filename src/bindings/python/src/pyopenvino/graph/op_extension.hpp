// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pybind11/pybind11.h>

#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "openvino/core/op_extension.hpp"

namespace py = pybind11;

class PyOpExtension : public ov::BaseOpExtension {
public:
    PyOpExtension(const py::object& dtype) : py_handle_dtype{dtype} {
        py::object type_info;
        try {
            // get_type_info() is a static method
            type_info = py_handle_dtype.attr("get_type_info")();
        } catch (const std::exception&) {
            try {
                //  get_type_info() is a class method
                type_info = py_handle_dtype().attr("get_type_info")();
            } catch (const std::exception&) {
                OPENVINO_THROW("Both options failed to get type_info.");
            }
        }
        if (!py::isinstance<ov::DiscreteTypeInfo>(type_info)) {
            OPENVINO_THROW("blahbla");
        }
        m_type_info = type_info.cast<std::shared_ptr<ov::DiscreteTypeInfo>>();
        OPENVINO_ASSERT(m_type_info->name != nullptr && m_type_info->version_id != nullptr,
                        "Extension type should have information about operation set and operation type.");
    }

    const ov::DiscreteTypeInfo& get_type_info() const override {
        return *m_type_info;
    }

    ov::OutputVector create(const ov::OutputVector& inputs, ov::AttributeVisitor& visitor) const override {
        // TODO: Create new python object using some python API under GIL then call its method
        py::gil_scoped_acquire acquire;
        // add check for default ctor
        const auto node = py_handle_dtype();

        node.attr("set_arguments")(py::cast(inputs));
        if (node.attr("visit_attributes")(&visitor)) {
            node.attr("constructor_validate_and_infer_types")();
        }

        return py::cast<ov::OutputVector>(node.attr("outputs")());
    }

    std::vector<ov::Extension::Ptr> get_attached_extensions() const override {
        return {};
    }

private:
    py::object py_handle_dtype;  // Holds the Python object to manage its lifetime
    std::shared_ptr<ov::DiscreteTypeInfo> m_type_info; 
};

void regclass_graph_OpExtension(py::module m);
