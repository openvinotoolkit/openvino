// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "common/frontend_exceptions.hpp"
#include "common/telemetry_extension.hpp"
#include "manager.hpp"
#include "pyopenvino/graph/function.hpp"

namespace py = pybind11;

using namespace ov::frontend;

void regclass_frontend_FrontEnd(py::module m) {
    py::class_<FrontEnd, std::shared_ptr<FrontEnd>> fem(m, "FrontEnd", py::dynamic_attr(), py::module_local());
    fem.doc() = "ngraph.impl.FrontEnd wraps ngraph::frontend::FrontEnd";

    fem.def(
        "load",
        [](FrontEnd& self, const std::string& s) {
            return self.load(s);
        },
        py::arg("path"),
        R"(
                Loads an input model by specified model file path.

                Parameters
                ----------
                path : str
                    Main model file path.

                Returns
                ----------
                load : InputModel
                    Loaded input model.
             )");

    fem.def("convert",
            static_cast<std::shared_ptr<ov::Model> (FrontEnd::*)(InputModel::Ptr) const>(&FrontEnd::convert),
            py::arg("model"),
            R"(
                Completely convert and normalize entire function, throws if it is not possible.

                Parameters
                ----------
                model : InputModel
                    Input model.

                Returns
                ----------
                convert : Model
                    Fully converted nGraph function.
             )");

    fem.def("convert",
            static_cast<void (FrontEnd::*)(std::shared_ptr<ov::Model>) const>(&FrontEnd::convert),
            py::arg("function"),
            R"(
                Completely convert the remaining, not converted part of a function.

                Parameters
                ----------
                function : Model
                    Partially converted nGraph function.

                Returns
                ----------
                convert : Model
                    Fully converted nGraph function.
             )");

    fem.def("convert_partially",
            &FrontEnd::convert_partially,
            py::arg("model"),
            R"(
                Convert only those parts of the model that can be converted leaving others as-is.
                Converted parts are not normalized by additional transformations; normalize function or
                another form of convert function should be called to finalize the conversion process.

                Parameters
                ----------
                model : InputModel
                    Input model.

                Returns
                ----------
                convert_partially : Model
                    Partially converted nGraph function.
             )");

    fem.def("decode",
            &FrontEnd::decode,
            py::arg("model"),
            R"(
                Convert operations with one-to-one mapping with decoding nodes.
                Each decoding node is an nGraph node representing a single FW operation node with
                all attributes represented in FW-independent way.

                Parameters
                ----------
                model : InputModel
                    Input model.

                Returns
                ----------
                decode : Model
                    nGraph function after decoding.
             )");

    fem.def("normalize",
            &FrontEnd::normalize,
            py::arg("function"),
            R"(
                Runs normalization passes on function that was loaded with partial conversion.

                Parameters
                ----------
                function : Model
                    Partially converted nGraph function.
             )");

    fem.def("get_name",
            &FrontEnd::get_name,
            R"(
                Gets name of this FrontEnd. Can be used by clients
                if frontend is selected automatically by FrontEndManager::load_by_model.

                Parameters
                ----------
                get_name : str
                    Current frontend name. Empty string if not implemented.
            )");

    fem.def("add_extension",
            static_cast<void (FrontEnd::*)(const std::shared_ptr<ov::Extension>& extension)>(&FrontEnd::add_extension));

    fem.def("__repr__", [](const FrontEnd& self) -> std::string {
        return "<FrontEnd '" + self.get_name() + "'>";
    });
}

void regclass_frontend_Extension(py::module m) {
    py::class_<ov::Extension, std::shared_ptr<ov::Extension>> ext(m, "Extension", py::dynamic_attr());
}

void regclass_frontend_TelemetryExtension(py::module m) {
    {
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
}
