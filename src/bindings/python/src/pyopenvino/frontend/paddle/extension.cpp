// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "extension.hpp"
#include "utils.hpp"

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "openvino/frontend/paddle/extension/conversion.hpp"
#include "openvino/frontend/paddle/extension/op.hpp"
#include "openvino/frontend/paddle/frontend.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace py = pybind11;

using namespace ov::frontend::paddle;

void regclass_frontend_paddle_ConversionExtension(py::module m) {
    py::class_<ConversionExtension, ConversionExtension::Ptr, ov::frontend::ConversionExtensionBase> _ext(
        m,
        "_ConversionExtensionPaddle",
        py::dynamic_attr());
    class PyConversionExtension : public ConversionExtension {
    public:
        using Ptr = std::shared_ptr<PyConversionExtension>;
        using PyCreatorFunctionNamed =
            std::function<std::map<std::string, ov::OutputVector>(const ov::frontend::NodeContext*)>;

        PyConversionExtension(const std::string& op_type, const PyCreatorFunctionNamed& f)
            : ConversionExtension(
                  op_type,
                  [f](const ov::frontend::NodeContext& node) -> std::map<std::string, ov::OutputVector> {
                      return f(static_cast<const ov::frontend::NodeContext*>(&node));
                  }) {}
    };
    py::class_<PyConversionExtension, PyConversionExtension::Ptr, ConversionExtension> ext(m,
                                                                                           "ConversionExtensionPaddle",
                                                                                           py::dynamic_attr());

    ext.def(py::init([](const std::string& op_type, const PyConversionExtension::PyCreatorFunctionNamed& f) {
        return std::make_shared<PyConversionExtension>(op_type, f);
    }));
}

void regclass_frontend_paddle_OpExtension(py::module m) {
    py::class_<OpExtension<void>, std::shared_ptr<OpExtension<void>>, ConversionExtension> ext(
            m,
            "OpExtensionPaddle",
            py::dynamic_attr());

    ext.def(py::init([](const std::string& fw_type_name,
                        const std::vector<std::string>& in_names_vec,
                        const std::vector<std::string>& out_names_vec,
                        const std::map<std::string, std::string>& attr_names_map,
                        const std::map<std::string, py::object>& attr_values_map) {

        std::map<std::string, ov::Any> any_map;
        for (const auto& it : attr_values_map) {
            any_map[it.first] = Common::utils::py_object_to_any(it.second);
        }
        return std::make_shared<OpExtension<void>>(fw_type_name, in_names_vec, out_names_vec, attr_names_map, any_map);
    }), py::arg("fw_type_name"),
            py::arg("in_names_vec"),
            py::arg("out_names_vec"),
            py::arg("attr_names_map") = std::map<std::string, std::string>(),
            py::arg("attr_values_map") = std::map<std::string, ov::Any>());

    ext.def(py::init([](const std::string& ov_type_name,
                               const std::string& fw_type_name,
                               const std::vector<std::string>& in_names_vec,
                               const std::vector<std::string>& out_names_vec,
                               const std::map<std::string, std::string>& attr_names_map,
                               const std::map<std::string, py::object>& attr_values_map) {

        std::map<std::string, ov::Any> any_map;
        for (const auto& it : attr_values_map) {
            any_map[it.first] = Common::utils::py_object_to_any(it.second);
        }
        return std::make_shared<OpExtension<void>>(ov_type_name, fw_type_name, in_names_vec, out_names_vec, attr_names_map, any_map);
    }),
            py::arg("ov_type_name"),
            py::arg("fw_type_name"),
            py::arg("in_names_vec"),
            py::arg("out_names_vec"),
            py::arg("attr_names_map") = std::map<std::string, std::string>(),
            py::arg("attr_values_map") = std::map<std::string, py::object>());
}
