// Copyright (C) 2018-2025 Intel Corporation
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
#include "openvino/util/file_util.hpp"
#include "pyopenvino/graph/model.hpp"
#include "pyopenvino/utils/utils.hpp"

namespace py = pybind11;

using namespace ov::frontend;

void regclass_frontend_FrontEnd(py::module m) {
    py::class_<FrontEnd, std::shared_ptr<FrontEnd>> fem(m, "FrontEnd", py::dynamic_attr(), py::module_local());
    fem.doc() = "openvino.frontend.FrontEnd wraps ov::frontend::FrontEnd";

    fem.def(py::init([](const std::shared_ptr<FrontEnd>& other) {
                return other;
            }),
            py::arg("other"));

    fem.def(
        "load",
        [](FrontEnd& self, const py::object& py_obj, const bool enable_mmap = true) {
            if (py::isinstance(py_obj, py::module_::import("pathlib").attr("Path")) ||
                py::isinstance<py::str>(py_obj) || py::isinstance<py::bytes>(py_obj)) {
                // check if model path is either a string/pathlib.Path/bytes
                std::string model_path = Common::utils::convert_path_to_string(py_obj);
                if (py::isinstance(py_obj, py::module_::import("pathlib").attr("Path")) ||
                    py::isinstance<py::str>(py_obj)) {

                // Fix unicode path
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
                    return self.load(ov::util::string_to_wstring(model_path.c_str()));
#else
                    return self.load(model_path.c_str());
#endif
                }
                return self.load(model_path, enable_mmap);
            } else if (py::isinstance(py_obj, pybind11::module::import("io").attr("BytesIO"))) {
                // support of BytesIO
                py::buffer_info info = py::buffer(py_obj.attr("getbuffer")()).request();
                Common::utils::MemoryBuffer mb(reinterpret_cast<char*>(info.ptr), info.size);
                std::istream _istream(&mb);
                return self.load(&_istream, enable_mmap);
            } else {
                // Extended for one argument only for this time
                return self.load({Common::utils::py_object_to_any(py_obj), enable_mmap});
            }
        },
        py::arg("path"),
        py::arg("enable_mmap") = true,
        R"(
                Loads an input model.

                :param path: Object describing the model. It can be path to model file.
                :type path: Any
                :param enable_mmap: Use mmap feature to map memory of a model's weights instead of reading directly. Optional. The default value is true.
                :type enable_mmap: boolean
                :return: Loaded input model.
                :rtype: openvino.frontend.InputModel
             )");

    fem.def(
        "supported",
        [](FrontEnd& self, const py::object& model) {
            if (py::isinstance(model, py::module_::import("pathlib").attr("Path")) || py::isinstance<py::str>(model) ||
                py::isinstance<py::bytes>(model)) {
                // check if model path is either a string/pathlib.Path/bytes
                std::string model_path = Common::utils::convert_path_to_string(model);
                if (py::isinstance(model, py::module_::import("pathlib").attr("Path")) ||
                    py::isinstance<py::str>(model)) {

                // Fix unicode path
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
                    return self.supported(ov::util::string_to_wstring(model_path.c_str()));
#else
                    return self.supported(model_path.c_str());
#endif
                }
            }
            return self.supported({Common::utils::py_object_to_any(model)});
        },
        py::arg("model"),
        R"(
                Checks if model type is supported.

                :param model: Object describing the model. It can be path to model file.
                :type model: Any
                :return: True if model type is supported, otherwise False.
                :rtype: bool
             )");

    fem.def("convert",
            static_cast<std::shared_ptr<ov::Model> (FrontEnd::*)(const InputModel::Ptr&) const>(&FrontEnd::convert),
            py::arg("model"),
            R"(
                Completely convert and normalize entire function, throws if it is not possible.

                :param model: Input model.
                :type model: openvino.frontend.InputModel
                :return: Fully converted OpenVINO Model.
                :rtype: openvino.Model
             )");

    fem.def(
        "convert",
        [](FrontEnd& self, const py::object& ie_api_model) {
            return self.convert(Common::utils::convert_to_model(ie_api_model));
        },
        py::arg("model"),
        R"(
                Completely convert the remaining, not converted part of a function.

                :param model: Partially converted OpenVINO model.
                :type model: openvino.frontend.Model
                :return: Fully converted OpenVINO Model.
                :rtype: openvino.Model
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
                :rtype: openvino.Model
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
                :rtype: openvino.Model
             )");

    fem.def(
        "normalize",
        [](FrontEnd& self, const py::object& ie_api_model) {
            self.normalize(Common::utils::convert_to_model(ie_api_model));
        },
        py::arg("model"),
        R"(
                Runs normalization passes on function that was loaded with partial conversion.

                :param model : Partially converted OpenVINO model.
                :type model: openvino.Model
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
            static_cast<void (FrontEnd::*)(const std::vector<std::shared_ptr<ov::Extension>>& extension)>(
                &FrontEnd::add_extension),
            R"(
                Add extensions defined by objects inheriting from Extension
                used in order to extend capabilities of Frontend.

                :param extension: Provided extension objects.
                :type extension: List[Extension]
            )");

    fem.def(
        "add_extension",
        [](FrontEnd& self, const py::object& extension_path) {
            return self.add_extension(Common::utils::convert_path_to_string(extension_path));
        },
        R"(
                Add extension defined in external library indicated by a extension_path
                used in order to extend capabilities of Frontend.

                :param extension_path: A path to extension.
                :type extension_path: str, Path
            )");

    fem.def("__repr__", [](const FrontEnd& self) -> std::string {
        return "<FrontEnd '" + self.get_name() + "'>";
    });
}
