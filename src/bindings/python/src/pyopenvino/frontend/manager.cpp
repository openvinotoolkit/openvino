// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/manager.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "openvino/frontend/exception.hpp"
#include "openvino/util/file_util.hpp"
#include "pyopenvino/frontend/manager.hpp"
#include "pyopenvino/utils/utils.hpp"

namespace py = pybind11;

void regclass_frontend_FrontEndManager(py::module m) {
    py::class_<ov::frontend::FrontEndManager, std::shared_ptr<ov::frontend::FrontEndManager>> fem(m,
                                                                                                  "FrontEndManager",
                                                                                                  py::dynamic_attr());
    fem.doc() = "openvino.frontend.FrontEndManager wraps ov::frontend::FrontEndManager";

    fem.def(py::init<>());

    // Empty pickle dumps are supported as FrontEndManager doesn't have any state
    fem.def(py::pickle(
        [](const ov::frontend::FrontEndManager&) {
            return py::make_tuple(0);
        },
        [](py::tuple t) {
            return ov::frontend::FrontEndManager();
        }));

    fem.def("get_available_front_ends",
            &ov::frontend::FrontEndManager::get_available_front_ends,
            R"(
                Gets list of registered frontends.

                :return: List of available frontend names.
                :rtype: List[str]
             )");

    fem.def(
        "register_front_end",
        [](const std::shared_ptr<ov::frontend::FrontEndManager>& fem,
           const std::string& name,
           const std::string& library_path) {
            return fem->register_front_end(name, library_path);
        },
        py::arg("name"),
        py::arg("library_path"),
        R"(
                Register frontend with name and factory loaded from provided library.

                :param name: Name of front end.
                :type name: str

                :param library_path: Path (absolute or relative) or name of a frontend library. If name is
                provided, depending on platform, it will be wrapped with shared library suffix and prefix
                to identify library full name.
                :type library_path: str

                :return: None
             )");

    fem.def("load_by_framework",
            &ov::frontend::FrontEndManager::load_by_framework,
            py::arg("framework"),
            R"(
                Loads frontend by name of framework and capabilities.

                :param framework: Framework name. Throws exception if name is not in list of available frontends.
                :type framework: str
                :return: Frontend interface for further loading of models.
                :rtype: openvino.frontend.FrontEnd
             )");

    fem.def(
        "load_by_model",
        [](const std::shared_ptr<ov::frontend::FrontEndManager>& fem, const py::object& model) {
            if (py::isinstance(model, py::module_::import("pathlib").attr("Path")) || py::isinstance<py::str>(model)) {
                std::string model_path = Common::utils::convert_path_to_string(model);

            // Fix unicode path
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
                return fem->load_by_model(ov::util::string_to_wstring(model_path.c_str()));
#else
                std::string model_path_str = model_path;
                return fem->load_by_model(model_path_str);
#endif
            }
            return fem->load_by_model({Common::utils::py_object_to_any(model)});
        },
        py::arg("model"),
        R"(
                Selects and loads appropriate frontend depending on model type or model file extension and other file info (header).

                :param model_path: A model object or path to a model file/directory.
                :type model_path: Any
                :return: Frontend interface for further loading of models. 'None' if no suitable frontend is found.
                :rtype: openvino.frontend.FrontEnd
            )");

    fem.def("__repr__", [](const ov::frontend::FrontEndManager& self) -> std::string {
        return "<FrontEndManager>";
    });
}

template <class T>
void handle_exception(py::module m, const char* exc_type) {
#if PYBIND11_VERSION_MAJOR < 2 || (PYBIND11_VERSION_MAJOR == 2 && PYBIND11_VERSION_MINOR < 12)
    static py::exception<T> exc(std::move(m), exc_type);
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p)
                std::rethrow_exception(p);
        } catch (const T& e) {
            exc(e.what());
        }
    });
#else
    static py::handle ex = py::exception<T>(std::move(m), exc_type).release();
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p)
                std::rethrow_exception(p);
        } catch (const T& e) {
            py::set_error(ex, e.what());
        }
    });
#endif
}

void regclass_frontend_GeneralFailureFrontEnd(py::module m) {
    handle_exception<ov::frontend::GeneralFailure>(std::move(m), "GeneralFailure");
}

void regclass_frontend_OpValidationFailureFrontEnd(py::module m) {
    handle_exception<ov::frontend::OpValidationFailure>(std::move(m), "OpValidationFailure");
}

void regclass_frontend_OpConversionFailureFrontEnd(py::module m) {
    handle_exception<ov::frontend::OpConversionFailure>(std::move(m), "OpConversionFailure");
}

void regclass_frontend_InitializationFailureFrontEnd(py::module m) {
    handle_exception<ov::frontend::InitializationFailure>(std::move(m), "InitializationFailure");
}

void regclass_frontend_NotImplementedFailureFrontEnd(py::module m) {
    handle_exception<ov::frontend::NotImplementedFailure>(std::move(m), "NotImplementedFailure");
}
