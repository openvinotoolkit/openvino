// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "extension/json_config.hpp"
#include "manager.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/frontend/extension/decoder_transformation.hpp"
#include "openvino/frontend/extension/progress_reporter_extension.hpp"
#include "openvino/frontend/extension/telemetry.hpp"
#include "pyopenvino/graph/model.hpp"

namespace py = pybind11;

using namespace ov::frontend;

void regclass_frontend_TelemetryExtension(py::module m) {
    py::class_<TelemetryExtension, std::shared_ptr<TelemetryExtension>, ov::Extension> ext(m,
                                                                                           "TelemetryExtension",
                                                                                           py::dynamic_attr());

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

void regclass_frontend_JsonConfigExtension(py::module m) {
    py::class_<ov::frontend::JsonConfigExtension,
               std::shared_ptr<ov::frontend::JsonConfigExtension>,
               ov::frontend::DecoderTransformationExtension>
        ext(m, "JsonConfigExtension", py::dynamic_attr());

    ext.doc() = "Extension class to load and process ModelOptimizer JSON config file";

    ext.def(py::init([](const std::string& path) {
        return std::make_shared<ov::frontend::JsonConfigExtension>(path);
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

    ext.def(py::init([](const ProgressReporterExtension::progress_notifier_callback& callback) {
        return std::make_shared<ProgressReporterExtension>(callback);
    }));

    ext.def(py::init([](ProgressReporterExtension::progress_notifier_callback&& callback) {
        return std::make_shared<ProgressReporterExtension>(std::move(callback));
    }));

    ext.def("report_progress", &ProgressReporterExtension::report_progress);
}
