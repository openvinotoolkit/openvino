// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/core/core.hpp"

#include <ie_extension.h>
#include <pybind11/stl.h>

#include <openvino/core/any.hpp>
#include <openvino/runtime/core.hpp>
#include <pyopenvino/core/tensor.hpp>
#include <pyopenvino/graph/any.hpp>

#include "common.hpp"

namespace py = pybind11;

using ConfigMap = std::map<std::string, std::string>;

std::string to_string(py::handle handle) {
    auto encodedString = PyUnicode_AsUTF8String(handle.ptr());
    return PyBytes_AsString(encodedString);
}

void regclass_Core(py::module m) {
    py::class_<ov::Core, std::shared_ptr<ov::Core>> cls(m, "Core");

    cls.def(py::init<const std::string&>(), py::arg("xml_config_file") = "");

    // todo: remove after Accuracy Checker migration to set/get_property API
    cls.def(
        "set_config",
        [](ov::Core& self, const std::map<std::string, std::string>& config, const std::string& device_name) {
            PyErr_WarnEx(PyExc_DeprecationWarning, "set_config() is deprecated, use set_property() instead.", 1);
            self.set_property(device_name, {config.begin(), config.end()});
        },
        py::arg("config"),
        py::arg("device_name") = "");

    cls.def(
        "set_property",
        [](ov::Core& self, const std::map<std::string, py::object>& properties) {
            std::map<std::string, PyAny> properties_to_cpp;
            for (const auto& property : properties) {
                properties_to_cpp[property.first] = PyAny(property.second);
            }
            self.set_property({properties_to_cpp.begin(), properties_to_cpp.end()});
        },
        py::arg("properties"),
        R"(
            Sets properties.
            Parameters
            ----------
            properties : dict
                Dict of pairs: (property name, property value)
            Returns
            ----------
            set_property : None
        )");

    cls.def(
        "set_property",
        [](ov::Core& self, const std::string& device_name, const std::map<std::string, py::object>& properties) {
            std::map<std::string, PyAny> properties_to_cpp;
            for (const auto& property : properties) {
                properties_to_cpp[property.first] = PyAny(property.second);
            }
            self.set_property(device_name, {properties_to_cpp.begin(), properties_to_cpp.end()});
        },
        py::arg("device_name"),
        py::arg("properties"),
        R"(
            Sets properties for the device.
            Parameters
            ----------
            device_name : str
                Name of the device to load the model to.
            properties : dict
                Dict of pairs: (property name, property value)
            Returns
            ----------
            set_property : None
        )");

    cls.def(
        "compile_model",
        [](ov::Core& self,
           const std::shared_ptr<const ov::Model>& model,
           const std::string& device_name,
           const std::map<std::string, std::string>& config) {
            return self.compile_model(model, device_name, {config.begin(), config.end()});
        },
        py::arg("model"),
        py::arg("device_name"),
        py::arg("config") = py::dict());

    cls.def(
        "compile_model",
        [](ov::Core& self,
           const std::shared_ptr<const ov::Model>& model,
           const std::map<std::string, std::string>& config) {
            return self.compile_model(model, ov::AnyMap{config.begin(), config.end()});
        },
        py::arg("model"),
        py::arg("config") = py::dict());

    cls.def(
        "compile_model",
        [](ov::Core& self,
           const std::string& model_path,
           const std::string& device_name,
           const std::map<std::string, std::string>& config) {
            return self.compile_model(model_path, device_name, {config.begin(), config.end()});
        },
        py::arg("model_path"),
        py::arg("device_name"),
        py::arg("config") = py::dict());

    cls.def(
        "compile_model",
        [](ov::Core& self, const std::string& model_path, const std::map<std::string, std::string>& config) {
            return self.compile_model(model_path, ov::AnyMap{config.begin(), config.end()});
        },
        py::arg("model_path"),
        py::arg("config") = py::dict());

    cls.def("get_versions", &ov::Core::get_versions, py::arg("device_name"));

    cls.def(
        "read_model",
        [](ov::Core& self, py::bytes model, py::bytes weights) {
            // works on view in order to omit copying bytes into string
            py::buffer_info info(py::buffer(weights).request());
            size_t bin_size = static_cast<size_t>(info.size);
            // if weights are not empty
            if (bin_size) {
                const uint8_t* bin = reinterpret_cast<const uint8_t*>(info.ptr);
                ov::Tensor tensor(ov::element::Type_t::u8, {bin_size});
                std::memcpy(tensor.data(), bin, bin_size);
                return self.read_model(model, tensor);
            }
            // create empty tensor of type u8
            ov::Tensor tensor(ov::element::Type_t::u8, {});
            return self.read_model(model, tensor);
        },
        py::arg("model"),
        py::arg("weights") = py::bytes());

    cls.def(
        "read_model",
        (std::shared_ptr<ov::Model>(ov::Core::*)(const std::string&, const std::string&) const) & ov::Core::read_model,
        py::arg("model"),
        py::arg("weights") = "");

    cls.def(
        "read_model",
        (std::shared_ptr<ov::Model>(ov::Core::*)(const std::string&, const ov::Tensor&) const) & ov::Core::read_model,
        py::arg("model"),
        py::arg("weights"));

    cls.def(
        "read_model",
        [](ov::Core& self, py::object model, py::object weights) {
            return self.read_model(py::str(model), py::str(weights));
        },
        py::arg("model"),
        py::arg("weights") = "");

    cls.def(
        "import_model",
        [](ov::Core& self,
           const std::string& model_stream,
           const std::string& device_name,
           const std::map<std::string, std::string>& properties) {
            std::stringstream _stream;
            _stream << model_stream;
            return self.import_model(_stream, device_name, {properties.begin(), properties.end()});
        },
        py::arg("model_stream"),
        py::arg("device_name"),
        py::arg("properties") = py::none(),
        R"(
            Imports a compiled model from a previously exported one.

            Parameters
            ----------
            model_stream : bytes
                Input stream containing a model previously exported using export_model method.

            device_name : str
                Name of device to import compiled model for.
                Note, if device_name device was not used to compile the original mode, an exception is thrown.

            properties : dict
                Optional map of pairs: (property name, property value) relevant only for this load operation.

            Returns
            ----------
            import_model : openvino.runtime.CompiledModel

            Examples
            ----------
            user_stream = compiled.export_model()

            with open('./my_model', 'wb') as f:
                f.write(user_stream)

            # ...

            new_compiled = core.import_model(user_stream, "CPU")
        )");

    // keep as second one to solve overload resolution problem
    cls.def(
        "import_model",
        [](ov::Core& self,
           const py::object& model_stream,
           const std::string& device_name,
           const std::map<std::string, std::string>& properties) {
            if (!(py::isinstance(model_stream, pybind11::module::import("io").attr("BytesIO")))) {
                throw py::type_error("CompiledModel.import_model(model_stream) incompatible function argument: "
                                     "`model_stream` must be an io.BytesIO object but " +
                                     (std::string)(py::repr(model_stream)) + "` provided");
            }
            model_stream.attr("seek")(0);  // Always rewind stream!
            std::stringstream _stream;
            _stream << model_stream
                           .attr("read")()  // alternative: model_stream.attr("get_value")()
                           .cast<std::string>();
            return self.import_model(_stream, device_name, {properties.begin(), properties.end()});
        },
        py::arg("model_stream"),
        py::arg("device_name"),
        py::arg("properties") = py::none(),
        R"(
            Imports a compiled model from a previously exported one.

            Advanced version of `import_model`. It utilizes, streams from standard
            Python library `io`.

            Parameters
            ----------
            model_stream : bytes
                Input stream containing a model previously exported using export_model method.

            device_name : str
                Name of device to import compiled model for.
                Note, if device_name device was not used to compile the original mode, an exception is thrown.

            properties : dict
                Optional map of pairs: (property name, property value) relevant only for this load operation.

            Returns
            ----------
            import_model : openvino.runtime.CompiledModel

            Examples
            ----------
            user_stream = io.BytesIO()
            compiled.export_model(user_stream)

            with open('./my_model', 'wb') as f:
                f.write(user_stream.getvalue()) # or read() if seek(0) was applied before

            # ...

            new_compiled = core.import_model(user_stream, "CPU")
        )");

    // todo: remove after Accuracy Checker migration to set/get_property API
    cls.def(
        "get_config",
        [](ov::Core& self, const std::string& device_name, const std::string& name) -> py::object {
            PyErr_WarnEx(PyExc_DeprecationWarning, "get_config() is deprecated, use get_property() instead.", 1);
            return Common::from_ov_any(self.get_property(device_name, name)).as<py::object>();
        },
        py::arg("device_name"),
        py::arg("name"));

    cls.def(
        "get_property",
        [](ov::Core& self, const std::string& device_name, const std::string& name) -> py::object {
            return Common::from_ov_any(self.get_property(device_name, name)).as<py::object>();
        },
        py::arg("device_name"),
        py::arg("name"),
        R"(
            Gets properties dedicated to device behaviour.
            Parameters
            ----------
            device_name : str
                A name of a device to get a properties value.
            name : str
                Property name.
            Returns
            ----------
            get_property : Any
        )");

    // todo: remove after Accuracy Checker migration to set/get_property API
    cls.def(
        "get_metric",
        [](ov::Core& self, const std::string device_name, const std::string name) -> py::object {
            PyErr_WarnEx(PyExc_DeprecationWarning, "get_metric() is deprecated, use get_property() instead.", 1);
            return Common::from_ov_any(self.get_property(device_name, name)).as<py::object>();
        },
        py::arg("device_name"),
        py::arg("name"));

    cls.def("register_plugin", &ov::Core::register_plugin, py::arg("plugin_name"), py::arg("device_name"));

    cls.def("register_plugins", &ov::Core::register_plugins, py::arg("xml_config_file"));

    cls.def("unload_plugin", &ov::Core::unload_plugin, py::arg("device_name"));

    cls.def(
        "query_model",
        [](ov::Core& self,
           const std::shared_ptr<const ov::Model>& model,
           const std::string& device_name,
           const std::map<std::string, std::string>& config) {
            return self.query_model(model, device_name, {config.begin(), config.end()});
        },
        py::arg("model"),
        py::arg("device_name"),
        py::arg("config") = py::dict());

    cls.def("add_extension",
            static_cast<void (ov::Core::*)(const std::string&)>(&ov::Core::add_extension),
            py::arg("library_path"));

    cls.def("add_extension",
            static_cast<void (ov::Core::*)(const std::shared_ptr<ov::Extension>&)>(&ov::Core::add_extension),
            py::arg("extension"));

    cls.def(
        "add_extension",
        static_cast<void (ov::Core::*)(const std::vector<std::shared_ptr<ov::Extension>>&)>(&ov::Core::add_extension),
        py::arg("extensions"));

    cls.def_property_readonly("available_devices", &ov::Core::get_available_devices);
}
