// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/frontend/extension.hpp"

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "openvino/frontend/exception.hpp"
#include "openvino/frontend/extension/conversion.hpp"
#include "openvino/frontend/extension/decoder_transformation.hpp"
#include "openvino/frontend/extension/op.hpp"
#include "openvino/frontend/extension/progress_reporter.hpp"
#include "openvino/frontend/extension/telemetry.hpp"
#include "pyopenvino/utils/utils.hpp"

namespace py = pybind11;

using namespace ov::frontend;

void regclass_frontend_TelemetryExtension(py::module m) {
    py::class_<TelemetryExtension, std::shared_ptr<TelemetryExtension>, ov::Extension> ext(m,
                                                                                           "TelemetryExtension",
                                                                                           py::dynamic_attr());

    ext.def(py::init([](const std::string& event_category,
                        py::function& send_event,
                        py::function& send_error,
                        py::function& send_stack_trace) {
        auto send_event_sp = Common::utils::wrap_pyfunction(send_event);
        auto send_error_sp = Common::utils::wrap_pyfunction(send_error);
        auto send_stack_trace_sp = Common::utils::wrap_pyfunction(send_stack_trace);

        return std::make_shared<TelemetryExtension>(
            event_category,
            [send_event_sp](const std::string& category,
                            const std::string& action,
                            const std::string& label,
                            int value) {
                py::gil_scoped_acquire acquire;
                (*send_event_sp)(category, action, label, value);
            },
            [send_error_sp](const std::string& category, const std::string& error_message) {
                py::gil_scoped_acquire acquire;
                (*send_error_sp)(category, error_message);
            },
            [send_stack_trace_sp](const std::string& category, const std::string& error_message) {
                py::gil_scoped_acquire acquire;
                (*send_stack_trace_sp)(category, error_message);
            });
    }));

    ext.def(py::init([](const std::string& event_category,
                        const TelemetryExtension::event_callback& send_event,
                        const TelemetryExtension::error_callback& send_error,
                        const TelemetryExtension::error_callback& send_stack_trace) {
        return std::make_shared<TelemetryExtension>(event_category, send_event, send_error, send_stack_trace);
    }));

    ext.def("send_event", &TelemetryExtension::send_event);
    ext.def("send_error", &TelemetryExtension::send_error);
    ext.def("send_stack_trace", &TelemetryExtension::send_stack_trace);
}

void regclass_frontend_DecoderTransformationExtension(py::module m) {
    py::class_<ov::frontend::DecoderTransformationExtension,
               std::shared_ptr<ov::frontend::DecoderTransformationExtension>,
               ov::Extension>
        ext(m, "DecoderTransformationExtension", py::dynamic_attr());
}

void regclass_frontend_ConversionExtensionBase(py::module m) {
    py::class_<ConversionExtensionBase, ConversionExtensionBase::Ptr, ov::Extension> ext(m,
                                                                                         "ConversionExtensionBase",
                                                                                         py::dynamic_attr());
}

void regclass_frontend_ConversionExtension(py::module m) {
    py::class_<ConversionExtension, ConversionExtension::Ptr, ConversionExtensionBase> _ext(m,
                                                                                            "_ConversionExtension",
                                                                                            py::dynamic_attr(),
                                                                                            py::module_local());
    class PyConversionExtension : public ConversionExtension {
    public:
        using Ptr = std::shared_ptr<PyConversionExtension>;
        using PyCreatorFunction = std::function<ov::OutputVector(const NodeContext*)>;
        using PyCreatorFunctionNamed = std::function<std::map<std::string, ov::OutputVector>(const NodeContext*)>;
        PyConversionExtension(const std::string& op_type, const PyCreatorFunction& f)
            : ConversionExtension(op_type, [f](const NodeContext& node) -> ov::OutputVector {
                  return f(static_cast<const NodeContext*>(&node));
              }) {}

        PyConversionExtension(const std::string& op_type, const PyCreatorFunctionNamed& f)
            : ConversionExtension(op_type, [f](const NodeContext& node) -> std::map<std::string, ov::OutputVector> {
                  return f(static_cast<const NodeContext*>(&node));
              }) {}
    };
    py::class_<PyConversionExtension, PyConversionExtension::Ptr, ConversionExtension> ext(m,
                                                                                           "ConversionExtension",
                                                                                           py::dynamic_attr());

    ext.def(py::init([](const std::string& op_type, const PyConversionExtension::PyCreatorFunction& f) {
        return std::make_shared<PyConversionExtension>(op_type, f);
    }));

    ext.def(py::init([](const std::string& op_type, const PyConversionExtension::PyCreatorFunctionNamed& f) {
        return std::make_shared<PyConversionExtension>(op_type, f);
    }));
}

void regclass_frontend_ProgressReporterExtension(py::module m) {
    py::class_<ProgressReporterExtension, std::shared_ptr<ProgressReporterExtension>, ov::Extension> ext{
        m,
        "ProgressReporterExtension",
        py::dynamic_attr()};

    ext.doc() = "An extension class intented to use as progress reporting utility";

    ext.def(py::init([]() {
        return std::make_shared<ProgressReporterExtension>();
    }));

    ext.def(py::init([](py::function& callback) {
        return std::make_shared<ProgressReporterExtension>([callback](float a, unsigned int b, unsigned int c) {
            py::gil_scoped_acquire acquire;
            callback(a, b, c);
        });
    }));

    ext.def(py::init([](const ProgressReporterExtension::progress_notifier_callback& callback) {
        return std::make_shared<ProgressReporterExtension>(callback);
    }));

    ext.def(py::init([](ProgressReporterExtension::progress_notifier_callback&& callback) {
        return std::make_shared<ProgressReporterExtension>(std::move(callback));
    }));

    ext.def("report_progress", &ProgressReporterExtension::report_progress);
}

void regclass_frontend_OpExtension(py::module m) {
    py::module frontend = m.def_submodule("frontend");
    py::class_<ov::frontend::OpExtension<void>, std::shared_ptr<ov::frontend::OpExtension<void>>, ConversionExtension>
        ext(frontend, "OpExtension", py::dynamic_attr());

    ext.def(py::init([](const std::string& fw_type_name,
                        const std::map<std::string, std::string>& attr_names_map,
                        const std::map<std::string, py::object>& attr_values_map) {
                std::map<std::string, ov::Any> any_map;
                for (const auto& it : attr_values_map) {
                    any_map[it.first] = Common::utils::py_object_to_any(it.second);
                }
                return std::make_shared<OpExtension<void>>(fw_type_name, attr_names_map, any_map);
            }),
            py::arg("fw_type_name"),
            py::arg("attr_names_map") = std::map<std::string, std::string>(),
            py::arg("attr_values_map") = std::map<std::string, py::object>());

    ext.def(py::init([](const std::string& ov_type_name,
                        const std::string& fw_type_name,
                        const std::map<std::string, std::string>& attr_names_map,
                        const std::map<std::string, py::object>& attr_values_map) {
                std::map<std::string, ov::Any> any_map;
                for (const auto& it : attr_values_map) {
                    any_map[it.first] = Common::utils::py_object_to_any(it.second);
                }

                return std::make_shared<OpExtension<void>>(ov_type_name, fw_type_name, attr_names_map, any_map);
            }),
            py::arg("ov_type_name"),
            py::arg("fw_type_name"),
            py::arg("attr_names_map") = std::map<std::string, std::string>(),
            py::arg("attr_values_map") = std::map<std::string, py::object>());

    ext.def(py::init([](const std::string& fw_type_name,
                        const std::vector<std::string>& in_names_vec,
                        const std::vector<std::string>& out_names_vec,
                        const std::map<std::string, std::string>& attr_names_map,
                        const std::map<std::string, py::object>& attr_values_map) {
                std::map<std::string, ov::Any> any_map;
                for (const auto& it : attr_values_map) {
                    any_map[it.first] = Common::utils::py_object_to_any(it.second);
                }
                return std::make_shared<OpExtension<void>>(fw_type_name,
                                                           in_names_vec,
                                                           out_names_vec,
                                                           attr_names_map,
                                                           any_map);
            }),
            py::arg("fw_type_name"),
            py::arg("in_names_vec"),
            py::arg("out_names_vec"),
            py::arg("attr_names_map") = std::map<std::string, std::string>(),
            py::arg("attr_values_map") = std::map<std::string, py::object>());

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

                return std::make_shared<OpExtension<void>>(ov_type_name,
                                                           fw_type_name,
                                                           in_names_vec,
                                                           out_names_vec,
                                                           attr_names_map,
                                                           any_map);
            }),
            py::arg("ov_type_name"),
            py::arg("fw_type_name"),
            py::arg("in_names_vec"),
            py::arg("out_names_vec"),
            py::arg("attr_names_map") = std::map<std::string, std::string>(),
            py::arg("attr_values_map") = std::map<std::string, py::object>());
}
