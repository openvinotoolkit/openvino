// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/compiled_model.hpp"

#include <pybind11/iostream.h>
#include <pybind11/stl.h>

#include "common.hpp"
#include "pyopenvino/core/containers.hpp"
#include "pyopenvino/core/infer_request.hpp"

PYBIND11_MAKE_OPAQUE(Containers::TensorIndexMap);
PYBIND11_MAKE_OPAQUE(Containers::TensorNameMap);

namespace py = pybind11;

void regclass_CompiledModel(py::module m) {
    py::class_<ov::CompiledModel, std::shared_ptr<ov::CompiledModel>> cls(m, "CompiledModel");
    cls.doc() = "openvino.runtime.CompiledModel represents Model that is compiled for a specific device by applying "
                "multiple optimization transformations, then mapping to compute kernels.";

    cls.def(py::init([](ov::CompiledModel& other) {
                return other;
            }),
            py::arg("other"));

    cls.def(
        "create_infer_request",
        [](ov::CompiledModel& self) {
            return std::make_shared<InferRequestWrapper>(self.create_infer_request(), self.inputs(), self.outputs());
        },
        R"(
            Creates an inference request object used to infer the compiled model.
            The created request has allocated input and output tensors.

            Parameters
            ----------
            None

            Returns
            ----------
            create_infer_request : openvino.runtime.InferRequest
                New InferRequest object.
        )");

    cls.def(
        "infer_new_request",
        [](ov::CompiledModel& self, const py::dict& inputs) {
            auto request = self.create_infer_request();
            // Update inputs if there are any
            Common::set_request_tensors(request, inputs);
            request.infer();
            return Common::outputs_to_dict(self.outputs(), request);
        },
        py::arg("inputs"),
        R"(
            Infers specified input(s) in synchronous mode.
            Blocks all methods of CompiledModel while request is running.

            Method creates new temporary InferRequest and run inference on it.
            It is advised to use dedicated InferRequest class for performance,
            optimizing workflows and creating advanced pipelines.

            Parameters
            ----------
            inputs : dict[Union[int, str, openvino.runtime.ConstOutput] : openvino.runtime.Tensor]
                Data to set on input tensors.

            Returns
            ----------
            infer_new_request : dict[openvino.runtime.ConstOutput : openvino.runtime.Tensor]
                Dictionary of results from output tensors with ports as keys.
        )");

    cls.def(
        "export_model",
        // &ov::CompiledModel::export_model,
        [](ov::CompiledModel& self, py::object model_stream) {
            // if (!(py::hasattr(fileHandle,"write") &&
            //     py::hasattr(fileHandle,"flush") )){
            // throw py::type_error("MyClass::read_from_file_like_object(file): incompatible function argument:  `file`
            // must be a file-like object, but `"
            //                             +(std::string)(py::repr(fileHandle))+"` provided"
            // );
            // }
            py::print("XD0");
            py::detail::pythonbuf buf(model_stream);
            py::print("XD1");
            std::ostream _stream(&buf);
            py::print("XD2");
            // self.export_model(_stream);
            py::scoped_ostream_redirect output{_stream};
            self.export_model(_stream);
        },
        // py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>(),
        py::arg("model_stream"),
        R"(
            Exports the compiled model to an output stream.

            Parameters
            ----------
            model_stream : str
                Stream where the model is going to be stored.

            Returns
            ----------
            export_model : None)
        )");

    cls.def(
        "get_config",
        [](ov::CompiledModel& self, const std::string& name) -> py::object {
            return Common::from_ov_any(self.get_property(name)).as<py::object>();
        },
        py::arg("name"));

    cls.def(
        "get_metric",
        [](ov::CompiledModel& self, const std::string& name) -> py::object {
            return Common::from_ov_any(self.get_property(name)).as<py::object>();
        },
        py::arg("name"));

    cls.def("get_runtime_model",
            &ov::CompiledModel::get_runtime_model,
            R"(
                Gets runtime model information from a device.

                This object (returned model) represents the internal device specific model
                which is optimized for particular accelerator. It contains device specific nodes,
                runtime information and can be used only to understand how the source model
                is optimized and which kernels, element types and layouts are selected.

                Parameters
                ----------
                None

                Returns
                ----------
                get_runtime_model : openvino.runtime.Model
                    Model containing Executable Graph information.
            )");

    cls.def_property_readonly("inputs",
                              &ov::CompiledModel::inputs,
                              R"(
                                Gets all inputs of a compiled model.

                                Returns
                                ----------
                                inputs : list[openvino.runtime.ConstOutput]
                              )");

    cls.def("input",
            (ov::Output<const ov::Node>(ov::CompiledModel::*)() const) & ov::CompiledModel::input,
            R"(
                Gets a single input of a compiled model.
                If a model has more than one input, this method throws an exception.

                Parameters
                ----------
                None

                Returns
                ----------
                input : openvino.runtime.ConstOutput
                    A compiled model input.
            )");

    cls.def("input",
            (ov::Output<const ov::Node>(ov::CompiledModel::*)(size_t) const) & ov::CompiledModel::input,
            py::arg("index"),
            R"(
                Gets input of a compiled model identified by an index.
                If an input with given index is not found, this method throws an exception.

                Parameters
                ----------
                index : int
                    An input index.

                Returns
                ----------
                input : openvino.runtime.ConstOutput
                    A compiled model input.
            )");

    cls.def("input",
            (ov::Output<const ov::Node>(ov::CompiledModel::*)(const std::string&) const) & ov::CompiledModel::input,
            py::arg("tensor_name"),
            R"(
                Gets input of a compiled model identified by a tensor_name.
                If an input with given tensor name is not found, this method throws an exception.

                Parameters
                ----------
                tensor_name : str
                    An input tensor's name.

                Returns
                ----------
                input : openvino.runtime.ConstOutput
                    A compiled model input.
            )");

    cls.def_property_readonly("outputs",
                              &ov::CompiledModel::outputs,
                              R"(
                                Gets all outputs of a compiled model.

                                Returns
                                ----------
                                outputs : list[openvino.runtime.ConstOutput]
                              )");

    cls.def("output",
            (ov::Output<const ov::Node>(ov::CompiledModel::*)() const) & ov::CompiledModel::output,
            R"(
                Gets a single output of a compiled model.
                If a model has more than one output, this method throws an exception.

                Parameters
                ----------
                None

                Returns
                ----------
                output : openvino.runtime.ConstOutput
                    A compiled model output.
            )");

    cls.def("output",
            (ov::Output<const ov::Node>(ov::CompiledModel::*)(size_t) const) & ov::CompiledModel::output,
            py::arg("index"),
            R"(
                Gets output of a compiled model identified by an index.
                If an output with given index is not found, this method throws an exception.

                Parameters
                ----------
                index : int
                    An output index.

                Returns
                ----------
                output : openvino.runtime.ConstOutput
                    A compiled model output.
            )");

    cls.def("output",
            (ov::Output<const ov::Node>(ov::CompiledModel::*)(const std::string&) const) & ov::CompiledModel::output,
            py::arg("tensor_name"),
            R"(
                Gets output of a compiled model identified by a tensor_name.
                If an output with given tensor name is not found, this method throws an exception.

                Parameters
                ----------
                tensor_name : str
                    An output tensor's name.

                Returns
                ----------
                output : openvino.runtime.ConstOutput
                    A compiled model output.
            )");
}
