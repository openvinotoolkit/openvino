// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/frontend/frontend.hpp"

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "openvino/frontend/exception.hpp"
#include "openvino/frontend/extension/telemetry.hpp"
#include "openvino/frontend/manager.hpp"
#include "pyopenvino/graph/model.hpp"
#include "pyopenvino/utils/utils.hpp"

namespace py = pybind11;

using namespace ov::frontend;

void regclass_frontend_FrontEnd(py::module m) {
    py::class_<FrontEnd, std::shared_ptr<FrontEnd>> fem(m, "FrontEnd", py::dynamic_attr(), py::module_local());
    fem.doc() = "openvino.frontend.FrontEnd wraps ov::frontend::FrontEnd";

    fem.def(
        "load",
        [](FrontEnd& self, const py::object& py_obj) {
            // TODO: Extend to arbitrary number of any parameters, this code shouldn't gate FE universal load function
            try {
                std::string model_path = Common::utils::convert_path_to_string(py_obj);
                return self.load(model_path);
            } catch (...) {
                // Extended for one argument only for this time
                return self.load({py_object_to_any(py_obj)});
            }
        },
        py::arg("py_obj"),
        R"(
                Loads an input model.

                :param py_obj: Object describing the model. It can be path to model file.
                :type py_obj: Any
                :return: Loaded input model.
                :rtype: openvino.frontend.InputModel
             )");

    fem.def("convert",
            static_cast<std::shared_ptr<ov::Model> (FrontEnd::*)(const InputModel::Ptr&) const>(&FrontEnd::convert),
            py::arg("model"),
            R"(
                Completely convert and normalize entire function, throws if it is not possible.

                :param model: Input model.
                :type model: openvino.frontend.InputModel
                :return: Fully converted OpenVINO Model.
                :rtype: openvino.runtime.Model
             )");

    fem.def("convert",
            static_cast<void (FrontEnd::*)(const std::shared_ptr<ov::Model>&) const>(&FrontEnd::convert),
            py::arg("model"),
            R"(
                Completely convert the remaining, not converted part of a function.

                :param model: Partially converted OpenVINO model.
                :type model: openvino.frontend.Model
                :return: Fully converted OpenVINO Model.
                :rtype: openvino.runtime.Model
             )");

    fem.def("convert_partially",
            &FrontEnd::convert_partially,
            py::arg("model"),
            R"(
                Convert only those parts of the model that can be converted leaving others as-is.
                Converted parts are not normalized by additional transformations; normalize function or
                another form of convert function should be called to finalize the conversion process.

                :param model : Input model.
                :type model: openvino.frontend.InputModel
                :return: Partially converted OpenVINO Model.
                :rtype: openvino.runtime.Model
             )");

    fem.def("decode",
            &FrontEnd::decode,
            py::arg("model"),
            R"(
                Convert operations with one-to-one mapping with decoding nodes.
                Each decoding node is an nGraph node representing a single FW operation node with
                all attributes represented in FW-independent way.

                :param model : Input model.
                :type model: openvino.frontend.InputModel
                :return: OpenVINO Model after decoding.
                :rtype: openvino.runtime.Model
             )");

    fem.def("normalize",
            &FrontEnd::normalize,
            py::arg("model"),
            R"(
                Runs normalization passes on function that was loaded with partial conversion.

                :param model : Partially converted OpenVINO model.
                :type model: openvino.runtime.Model
             )");

    fem.def("get_name",
            &FrontEnd::get_name,
            R"(
                Gets name of this FrontEnd. Can be used by clients
                if frontend is selected automatically by FrontEndManager::load_by_model.

                :return: Current frontend name. Returns empty string if not implemented.
                :rtype: str
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
