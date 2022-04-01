// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/compiled_model.hpp"

#include <pybind11/iostream.h>
#include <pybind11/stl.h>

#include "common.hpp"
#include "pyopenvino/core/containers.hpp"
#include "pyopenvino/core/infer_request.hpp"
#include "pyopenvino/utils/utils.hpp"

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
        py::call_guard<py::gil_scoped_release>(),
        R"(
            Creates an inference request object used to infer the compiled model.
            The created request has allocated input and output tensors.

            :return: New InferRequest object.
            :rtype: openvino.runtime.InferRequest
        )");

    cls.def(
        "infer_new_request",
        [](ov::CompiledModel& self, const py::dict& inputs) {
            auto request = self.create_infer_request();
            // Update inputs if there are any
            Common::set_request_tensors(request, inputs);
            {
                py::gil_scoped_release release;
                request.infer();
            }
            return Common::outputs_to_dict(self.outputs(), request);
        },
        py::arg("inputs"),
        R"(
            Infers specified input(s) in synchronous mode.
            Blocks all methods of CompiledModel while the request is running.

            Method creates new temporary InferRequest and run inference on it.
            It is advised to use a dedicated InferRequest class for performance,
            optimizing workflows, and creating advanced pipelines.

            GIL is released during the inference.

            :param inputs: Data to set on input tensors.
            :type inputs: Dict[Union[int, str, openvino.runtime.ConstOutput], openvino.runtime.Tensor]
            :return: Dictionary of results from output tensors with ports as keys.
            :rtype: Dict[openvino.runtime.ConstOutput, numpy.array]
        )");

    cls.def(
        "export_model",
        [](ov::CompiledModel& self) {
            std::stringstream _stream;
            self.export_model(_stream);
            return py::bytes(_stream.str());
        },
        py::call_guard<py::gil_scoped_release>(),
        R"(
            Exports the compiled model to bytes/output stream.

            GIL is released while running this function.

            :return: Bytes object that contains this compiled model.
            :rtype: bytes

            .. code-block:: python

                user_stream = compiled.export_model()

                with open('./my_model', 'wb') as f:
                    f.write(user_stream)

                # ...

                new_compiled = core.import_model(user_stream, "CPU")
        )");

    cls.def(
        "export_model",
        [](ov::CompiledModel& self, py::object& model_stream) {
            if (!(py::isinstance(model_stream, pybind11::module::import("io").attr("BytesIO")))) {
                throw py::type_error("CompiledModel.export_model(model_stream) incompatible function argument: "
                                     "`model_stream` must be an io.BytesIO object but " +
                                     (std::string)(py::repr(model_stream)) + "` provided");
            }
            std::stringstream _stream;
            {
                py::gil_scoped_release release;
                self.export_model(_stream);
            }
            model_stream.attr("flush")();
            model_stream.attr("write")(py::bytes(_stream.str()));
            model_stream.attr("seek")(0);  // Always rewind stream!
        },
        py::arg("model_stream"),
        R"(
            Exports the compiled model to bytes/output stream.

            Advanced version of `export_model`. It utilizes, streams from the standard
            Python library `io`.

            Function performs flushing of the stream, writes to it, and then rewinds
            the stream to the beginning (using seek(0)).

            GIL is released while running this function.

            :param model_stream: A stream object to which the model will be serialized.
            :type model_stream: io.BytesIO
            :rtype: None

            .. code-block:: python

                user_stream = io.BytesIO()
                compiled.export_model(user_stream)

                with open('./my_model', 'wb') as f:
                    f.write(user_stream.getvalue()) # or read() if seek(0) was applied before

                # ...

                new_compiled = core.import_model(user_stream, "CPU")
        )");

    cls.def(
        "set_property",
        [](ov::CompiledModel& self, const std::map<std::string, py::object>& properties) {
            std::map<std::string, ov::Any> properties_to_cpp;
            for (const auto& property : properties) {
                properties_to_cpp[property.first] = ov::Any(py_object_to_any(property.second));
            }
            self.set_property({properties_to_cpp.begin(), properties_to_cpp.end()});
        },
        py::arg("properties"),
        R"(
            Sets properties for current compiled model.

            :param properties: Dict of pairs: (property name, property value)
            :type properties: dict
            :rtype: None
        )");

    cls.def(
        "get_property",
        [](ov::CompiledModel& self, const std::string& name) -> py::object {
            return Common::utils::from_ov_any(self.get_property(name));
        },
        py::arg("name"),
        R"(
            Gets properties for current compiled model.

            :param name: Property name.
            :type name: str
            :rtype: Any
        )");

    cls.def("get_runtime_model",
            &ov::CompiledModel::get_runtime_model,
            py::call_guard<py::gil_scoped_release>(),
            R"(
                Gets runtime model information from a device.

                This object (returned model) represents the internal device-specific model
                which is optimized for the particular accelerator. It contains device-specific nodes,
                runtime information, and can be used only to understand how the source model
                is optimized and which kernels, element types, and layouts are selected.

                :return: Model, containing Executable Graph information.
                :rtype: openvino.runtime.Model
            )");

    cls.def_property_readonly("inputs",
                              &ov::CompiledModel::inputs,
                              R"(
                                Gets all inputs of a compiled model.

                                :return: Inputs of a compiled model.
                                :rtype: List[openvino.runtime.ConstOutput]
                              )");

    cls.def("input",
            (ov::Output<const ov::Node>(ov::CompiledModel::*)() const) & ov::CompiledModel::input,
            R"(
                Gets a single input of a compiled model.
                If a model has more than one input, this method throws an exception.

                :return: A compiled model input.
                :rtype: openvino.runtime.ConstOutput
            )");

    cls.def("input",
            (ov::Output<const ov::Node>(ov::CompiledModel::*)(size_t) const) & ov::CompiledModel::input,
            py::arg("index"),
            R"(
                Gets input of a compiled model identified by an index.
                If the input with given index is not found, this method throws an exception.

                :param index: An input index.
                :type index: int
                :return: A compiled model input.
                :rtype: openvino.runtime.ConstOutput
            )");

    cls.def("input",
            (ov::Output<const ov::Node>(ov::CompiledModel::*)(const std::string&) const) & ov::CompiledModel::input,
            py::arg("tensor_name"),
            R"(
                Gets input of a compiled model identified by a tensor_name.
                If the input with given tensor name is not found, this method throws an exception.

                :param tensor_name: An input tensor name.
                :type tensor_name: str
                :return: A compiled model input.
                :rtype: openvino.runtime.ConstOutput
            )");

    cls.def_property_readonly("outputs",
                              &ov::CompiledModel::outputs,
                              R"(
                                Gets all outputs of a compiled model.

                                :return: Outputs of a compiled model.
                                :rtype: List[openvino.runtime.ConstOutput]
                              )");

    cls.def("output",
            (ov::Output<const ov::Node>(ov::CompiledModel::*)() const) & ov::CompiledModel::output,
            R"(
                Gets a single output of a compiled model.
                If the model has more than one output, this method throws an exception.

                :return: A compiled model output.
                :rtype: openvino.runtime.ConstOutput
            )");

    cls.def("output",
            (ov::Output<const ov::Node>(ov::CompiledModel::*)(size_t) const) & ov::CompiledModel::output,
            py::arg("index"),
            R"(
                Gets output of a compiled model identified by an index.
                If the output with given index is not found, this method throws an exception.

                :param index: An output index.
                :type index: int
                :return: A compiled model output.
                :rtype: openvino.runtime.ConstOutput
            )");

    cls.def("output",
            (ov::Output<const ov::Node>(ov::CompiledModel::*)(const std::string&) const) & ov::CompiledModel::output,
            py::arg("tensor_name"),
            R"(
                Gets output of a compiled model identified by a tensor_name.
                If the output with given tensor name is not found, this method throws an exception.

                :param tensor_name: An output tensor name.
                :type tensor_name: str
                :return: A compiled model output.
                :rtype: openvino.runtime.ConstOutput
            )");

    cls.def("__repr__", [](const ov::CompiledModel& self) {
        auto inputs_str = Common::docs::container_to_string(self.inputs(), ",\n");
        auto outputs_str = Common::docs::container_to_string(self.outputs(), ",\n");

        return "<CompiledModel:\ninputs[\n" + inputs_str + "\n]\noutputs[\n" + outputs_str + "\n]>";
    });
}
