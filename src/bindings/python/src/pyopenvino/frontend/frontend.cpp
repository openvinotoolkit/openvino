// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "openvino/frontend/exception.hpp"
#include "openvino/frontend/extension/telemetry.hpp"
#include "openvino/frontend/manager.hpp"
#include "pyopenvino/graph/model.hpp"

namespace py = pybind11;

using namespace ov::frontend;

void regclass_frontend_FrontEnd(py::module m) {
    py::class_<FrontEnd, std::shared_ptr<FrontEnd>> fem(m, "FrontEnd", py::dynamic_attr(), py::module_local());
    fem.doc() = "openvino.frontend.FrontEnd wraps ov::frontend::FrontEnd";

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
            static_cast<std::shared_ptr<ov::Model> (FrontEnd::*)(const InputModel::Ptr&) const>(&FrontEnd::convert),
            py::arg("model"),
            py::keep_alive<0, 1>(),
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
            static_cast<void (FrontEnd::*)(const std::shared_ptr<ov::Model>&) const>(&FrontEnd::convert),
            py::arg("function"),
            R"(
                Completely convert the remaining, not converted part of a function.

                Parameters
                ----------
                function : Model
                    Partially converted nGraph function.
             )");

    fem.def("convert_partially",
            &FrontEnd::convert_partially,
            py::arg("model"),
            py::keep_alive<0, 1>(),
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
            py::keep_alive<0, 1>(),
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
            static_cast<void (FrontEnd::*)(const std::shared_ptr<ov::Extension>& extension)>(&FrontEnd::add_extension),
            R"(
                Add extension defined by an object inheriting from Extension 
                used in order to extend capabilities of Frontend.

                :param extension: Provided extension object.
                :type extension: Extension
            )");

    fem.def("add_extension",
            static_cast<void (FrontEnd::*)(const std::string& extension_path)>(&FrontEnd::add_extension),
            R"(
                Add extension defined in external library indicated by a extension_path 
                used in order to extend capabilities of Frontend.

                :param extension_path: A path to extension.
                :type extension_path: str
            )");

    fem.def("__repr__", [](const FrontEnd& self) -> std::string {
        return "<FrontEnd '" + self.get_name() + "'>";
    });
}
