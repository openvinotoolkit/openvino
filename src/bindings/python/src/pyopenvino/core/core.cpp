// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/core/core.hpp"

#include <ie_extension.h>
#include <pybind11/stl.h>

#include <openvino/core/any.hpp>
#include <openvino/runtime/core.hpp>
#include <pyopenvino/core/tensor.hpp>

#include "common.hpp"

namespace py = pybind11;

using ConfigMap = std::map<std::string, std::string>;

std::string to_string(py::handle handle) {
    auto encodedString = PyUnicode_AsUTF8String(handle.ptr());
    return PyBytes_AsString(encodedString);
}

void regclass_Core(py::module m) {
    py::class_<ov::Core, std::shared_ptr<ov::Core>> cls(m, "Core");
    cls.doc() =
        "openvino.runtime.Core class represents OpenVINO runtime Core entity. User applications can create several "
        "Core class instances, but in this case the underlying plugins are created multiple times and not shared "
        "between several Core instances. The recommended way is to have a single Core instance per application.";

    cls.def(py::init<const std::string&>(), py::arg("xml_config_file") = "");

    cls.def(
        "set_property",
        [](ov::Core& self, const std::string& device_name, const std::map<std::string, std::string>& config) {
            self.set_property(device_name, {config.begin(), config.end()});
        },
        py::arg("device_name") = "",
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
           const std::map<std::string, std::string>& properties) {
            return self.compile_model(model, device_name, {properties.begin(), properties.end()});
        },
        py::arg("model"),
        py::arg("device_name"),
        py::arg("config") = py::dict(),
        R"(
            Creates a compiled model from a source model object.
            Users can create as many compiled models as they need and use them simultaneously
            (up to the limitation of the hardware resources).

            Parameters
            ----------
            model : openvino.runtime.Model
                Model acquired from read_model function.

            device_name : str
                Name of the device to load the model to.

            properties : dict
                Optional dict of pairs: (property name, property value) relevant only for this load operation.

            Returns
            ----------
            compile_model : openvino.runtime.CompiledModel
        )");

    cls.def(
        "compile_model",
        [](ov::Core& self,
           const std::shared_ptr<const ov::Model>& model,
           const std::map<std::string, std::string>& config) {
            return self.compile_model(model, ov::AnyMap{config.begin(), config.end()});
        },
        py::arg("model"),
        py::arg("config") = py::dict(),
        R"(
            Creates and loads a compiled model from a source model to the default OpenVINO device
            selected by AUTO plugin. Users can create as many compiled models as they need and use
            them simultaneously (up to the limitation of the hardware resources).


            Parameters
            ----------
            model : openvino.runtime.Model
                Model acquired from read_model function.

            properties : dict
                Optional dict of pairs: (property name, property value) relevant only for this load operation.

            Returns
            ----------
            compile_model : openvino.runtime.CompiledModel
        )");

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
        py::arg("properties") = py::dict(),
        R"(
            Reads model and creates a compiled model from IR / ONNX / PDPD file.
            This can be more efficient than using read_model + compile_model(model_in_memory_object) flow,
            especially for cases when caching is enabled and cached model is available.


            Parameters
            ----------
            model_path : str
                A path to a model in IR / ONNX / PDPD format.

            device_name : str
                Name of the device to load the model to.

            properties : dict
                Optional dict of pairs: (property name, property value) relevant only for this load operation.

            Returns
            ----------
            compile_model : openvino.runtime.CompiledModel
        )");

    cls.def(
        "compile_model",
        [](ov::Core& self, const std::string& model_path, const std::map<std::string, std::string>& properties) {
            return self.compile_model(model_path, ov::AnyMap{properties.begin(), properties.end()});
        },
        py::arg("model_path"),
        py::arg("config") = py::dict(),
        R"(
            Reads model and creates a compiled model from IR / ONNX / PDPD file with device selected by AUTO plugin.
            This can be more efficient than using read_model + compile_model(model_in_memory_object) flow,
            especially for cases when caching is enabled and cached model is available.

            Parameters
            ----------
            model_path : str
                A path to a model in IR / ONNX / PDPD format.

            properties : dict
                Optional dict of pairs: (property name, property value) relevant only for this load operation.

            Returns
            ----------
            compile_model : openvino.runtime.CompiledModel
        )");

    cls.def("get_versions",
            &ov::Core::get_versions,
            py::arg("device_name"),
            R"(
                Returns device plugins version information.

                Parameters
                ----------
                device_name : str
                    Device name to identify a plugin.

                Returns
                ----------
                get_versions : dict[str, openvino.runtime.Version]
            )");

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
        py::arg("weights") = py::bytes(),
        R"(
            Reads models from IR / ONNX / PDPD formats.

            Parameters
            ----------
            model : bytes
                Bytes with model in IR / ONNX / PDPD format.

            weights : bytes
                Bytes with tensor's data.

            Returns
            ----------
            read_model : openvino.runtime.Model
        )");

    cls.def(
        "read_model",
        (std::shared_ptr<ov::Model>(ov::Core::*)(const std::string&, const std::string&) const) & ov::Core::read_model,
        py::arg("model"),
        py::arg("weights") = "",
        R"(
            Reads models from IR / ONNX / PDPD formats.

            Parameters
            ----------
            model : str
                A path to a model in IR / ONNX / PDPD format.

            weights : str
                A path to a data file For IR format (*.bin): if path is empty,
                will try to read bin file with the same name as xml and if bin
                file with the same name was not found, will load IR without weights.
                For ONNX format (*.onnx): weights parameter is not used.
                For PDPD format (*.pdmodel) weights parameter is not used.

            Returns
            ----------
            read_model : openvino.runtime.Model
        )");

    cls.def(
        "read_model",
        (std::shared_ptr<ov::Model>(ov::Core::*)(const std::string&, const ov::Tensor&) const) & ov::Core::read_model,
        py::arg("model"),
        py::arg("weights"),
        R"(
            Reads models from IR / ONNX / PDPD formats.

            Parameters
            ----------
            model : str
                A string with model in IR / ONNX / PDPD format.

            weights : openvino.runtime.Tensor
                Tensor with weights. Reading ONNX / PDPD models doesn't support loading weights from weights tensors.

            Returns
            ----------
            read_model : openvino.runtime.Model
        )");

    cls.def(
        "read_model",
        [](ov::Core& self, py::object model, py::object weights) {
            return self.read_model(py::str(model), py::str(weights));
        },
        py::arg("model"),
        py::arg("weights") = "",
        R"(
            Reads models from IR / ONNX / PDPD formats.

            Parameters
            ----------
            model : str
                A string with model in IR / ONNX / PDPD format.

            weights : str
                A path to a data file For IR format (*.bin): if path is empty,
                will try to read bin file with the same name as xml and if bin
                file with the same name was not found, will load IR without weights.
                For ONNX format (*.onnx): weights parameter is not used.
                For PDPD format (*.pdmodel) weights parameter is not used.

            Returns
            ----------
            read_model : openvino.runtime.Model
        )");

    cls.def(
        "import_model",
        [](ov::Core& self,
           std::istream& model_file,
           const std::string& device_name,
           const std::map<std::string, std::string>& properties) {
            return self.import_model(model_file, device_name, {properties.begin(), properties.end()});
        },
        py::arg("model_file"),
        py::arg("device_name"),
        py::arg("properties") = py::none(),
        R"(
            Imports a compiled model from a previously exported one.

            Parameters
            ----------
            model_file : str //jiwaszki model_stream but how
                Input stream containing a model previously exported using export_model method.

            device_name : str
                Name of device to import compiled model for.
                Note, if device_name device was not used to compile the original mode, an exception is thrown.

            properties : dict
                Optional map of pairs: (property name, property value) relevant only for this load operation.

            Returns
            ----------
            import_model : openvino.runtime.Model
        )");

    cls.def(
        "get_config",
        [](ov::Core& self, const std::string& device_name, const std::string& name) -> py::object {
            return Common::from_ov_any(self.get_property(device_name, name)).as<py::object>();
        },
        py::arg("device_name"),
        py::arg("name"));

    cls.def(
        "get_metric",
        [](ov::Core& self, const std::string device_name, const std::string name) -> py::object {
            return Common::from_ov_any(self.get_property(device_name, name)).as<py::object>();
        },
        py::arg("device_name"),
        py::arg("name"));

    cls.def("register_plugin",
            &ov::Core::register_plugin,
            py::arg("plugin_name"),
            py::arg("device_name"),
            R"(
                Register a new device and plugin which enable this device inside OpenVINO Runtime.

                Parameters
                ----------
                plugin_name : str
                    A name of plugin. Depending on platform `plugin_name` is wrapped with shared library
                    suffix and prefix to identify library full name E.g. on Linux platform plugin name
                    specified as `plugin_name` will be wrapped as `libplugin_name.so`.

                device_name : str
                    A device name to register plugin for.

                Returns
                ----------
                register_plugin : None
            )");

    cls.def("register_plugins",
            &ov::Core::register_plugins,
            py::arg("xml_config_file"),
            R"(
                Registers a device plugin to OpenVINO Runtime Core instance using XML configuration
                file with plugins description.

                Parameters
                ----------
                xml_config_file : str
                    A path to .xml file with plugins to register.

                Returns
                ----------
                register_plugins : None
            )");

    cls.def("unload_plugin",
            &ov::Core::unload_plugin,
            py::arg("device_name"),
            R"(
                Unloads the previously loaded plugin identified by device_name from OpenVINO Runtime.
                The method is needed to remove loaded plugin instance and free its resources.
                If plugin for a specified device has not been created before, the method throws an exception.

                Parameters
                ----------
                device_name : str
                    A device name identifying plugin to remove from OpenVINO.

                Returns
                ----------
                unload_plugin : None
            )");

    cls.def(
        "query_model",
        [](ov::Core& self,
           const std::shared_ptr<const ov::Model>& model,
           const std::string& device_name,
           const std::map<std::string, std::string>& properties) {
            return self.query_model(model, device_name, {properties.begin(), properties.end()});
        },
        py::arg("model"),
        py::arg("device_name"),
        py::arg("properties") = py::dict(),
        R"(
            Query device if it supports specified model with specified properties.

            Parameters
            ----------
            model : openvino.runtime.Model
                Model object to query.

            device_name : str
                A name of a device to query.

            properties : dict
                Optional dict of pairs: (property name, property value)

            Returns
            ----------
            query_model : dict
                Pairs a operation name -> a device name supporting this operation.
        )");

    cls.def("add_extension",
            static_cast<void (ov::Core::*)(const std::string&)>(&ov::Core::add_extension),
            py::arg("library_path"),
            R"(
                Registers an extension to a Core object.

                Parameters
                ----------
                library_path : str
                    Path to library with ov::Extension

                Returns
                ----------
                add_extension : None
            )");

    cls.def("add_extension",
            static_cast<void (ov::Core::*)(const std::shared_ptr<ov::Extension>&)>(&ov::Core::add_extension),
            py::arg("extension"),
            R"(
                Registers an extension to a Core object.

                Parameters
                ----------
                extension : openvino.runtime.Extension
                    Extension object.

                Returns
                ----------
                add_extension : None
            )");

    cls.def(
        "add_extension",
        static_cast<void (ov::Core::*)(const std::vector<std::shared_ptr<ov::Extension>>&)>(&ov::Core::add_extension),
        py::arg("extensions"),
        R"(
            Registers extensions to a Core object.

            Parameters
            ----------
            extensions : list[openvino.runtime.Extension]
                List of Extension objects.

            Returns
            ----------
            add_extension : None
        )");

    cls.def_property_readonly("available_devices",
                              &ov::Core::get_available_devices,
                              R"(
                                    Returns devices available for inference Core objects goes over all registered plugins.

                                    Returns
                                    ----------
                                    available_devices : list
                                        A list of devices. The devices are returned as: CPU, GPU.0, GPU.1, MYRIAD...
                                        If there more than one device of specific type, they are enumerated with .# suffix.
                                        Such enumerated device can later be used as a device name in all Core methods like:
                                        compile_model, query_model, set_property and so on.
                                )");
}
