// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/graph/preprocess/pre_post_process.hpp"

#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include "openvino/core/function.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"

namespace py = pybind11;

static void regclass_graph_PreProcessSteps(py::module m) {
    py::class_<ov::preprocess::PreProcessSteps, std::shared_ptr<ov::preprocess::PreProcessSteps>> steps(
        m,
        "PreProcessSteps");
    steps.doc() = "openvino.impl.preprocess.PreProcessSteps wraps ov::preprocess::PreProcessSteps";

    steps.def(py::init<>());

    steps.def(
        "mean",
        [](const std::shared_ptr<ov::preprocess::PreProcessSteps>& me, float value) {
            me->mean(value);
            return me;
        },
        py::arg("value"),
        R"(
                Subtracts single float value from each element in input tensor.
                Input tensor must have ov.Type.f32 data type.
                Parameters
                ----------
                value : float
                    Value to subtract.
                Returns
                ----------
                mean : PreProcessSteps
                    Reference to itself to allow chaining of calls in client's code in a builder-like manner.
              )");
    steps.def(
        "mean",
        [](const std::shared_ptr<ov::preprocess::PreProcessSteps>& me, const std::vector<float>& values) {
            me->mean(values);
            return me;
        },
        py::arg("values"),
        R"(
                Subtracts a given single float value from each element in a given channel from input tensor.
                Input tensor must have ov.Type.f32 data type.
                Parameters
                ----------
                values : List
                    Values to subtract.
                Returns
                ----------
                mean : PreProcessSteps
                    Reference to itself to allow chaining of calls in client's code in a builder-like manner.
              )");
    steps.def(
        "scale",
        [](const std::shared_ptr<ov::preprocess::PreProcessSteps>& me, float value) {
            me->scale(value);
            return me;
        },
        py::arg("value"),
        R"(
                Divides each element in input tensor by specified constant float value.
                Input tensor must have ov.Type.f32 data type.
                Parameters
                ----------
                value : float
                    Value to divide.
                Returns
                ----------
                scale : PreProcessSteps
                    Reference to itself to allow chaining of calls in client's code in a builder-like manner.
              )");
    steps.def(
        "scale",
        [](const std::shared_ptr<ov::preprocess::PreProcessSteps>& me, const std::vector<float>& values) {
            me->scale(values);
            return me;
        },
        py::arg("values"),
        R"(
                Divides each element in a given channel from input tensor by a given single float value.
                Input tensor must have ov.Type.f32 data type.
                Parameters
                ----------
                value : List
                    Value to divide.
                Returns
                ----------
                scale : PreProcessSteps
                    Reference to itself to allow chaining of calls in client's code in a builder-like manner.
              )");
    steps.def(
        "convert_element_type",
        [](const std::shared_ptr<ov::preprocess::PreProcessSteps>& me, ov::element::Type type) {
            me->convert_element_type(type);
            return me;
        },
        py::arg("type"),
        R"(
                Converts input tensor element type to specified type.
                Input tensor must have openvino.Type.f32 data type.
                Parameters
                ----------
                type : Type
                    Destination type.
                Returns
                ----------
                convert_element_type : PreProcessSteps
                    Reference to itself to allow chaining of calls in client's code in a builder-like manner.
              )");
    steps.def(
        "custom",
        [](const std::shared_ptr<ov::preprocess::PreProcessSteps>& me, py::function op) {
            me->custom(op.cast<const ov::preprocess::PreProcessSteps::CustomPreprocessOp>());
            return me;
        },
        py::arg("operation"),
        R"(
                Adds custom preprocessing operation.
                Parameters
                ----------
                operation : function taking Output<Node> as input argument and returning Output<Node> after preprocessing.
                Returns
                ----------
                custom : PreProcessSteps
                    Reference to itself to allow chaining of calls in client's code in a builder-like manner.
              )");
    steps.def(
        "convert_color",
        [](const std::shared_ptr<ov::preprocess::PreProcessSteps>& me, const ov::preprocess::ColorFormat& dst_format) {
            me->convert_color(dst_format);
            return me;
        },
        py::arg("dst_format"));
    steps.def(
        "resize",
        [](const std::shared_ptr<ov::preprocess::PreProcessSteps>& me,
           ov::preprocess::ResizeAlgorithm alg,
           size_t dst_height,
           size_t dst_width) {
            me->resize(alg, dst_height, dst_width);
            return me;
        },
        py::arg("alg"),
        py::arg("dst_height"),
        py::arg("dst_width"));
    steps.def(
        "resize",
        [](const std::shared_ptr<ov::preprocess::PreProcessSteps>& me, ov::preprocess::ResizeAlgorithm alg) {
            me->resize(alg);
            return me;
        },
        py::arg("alg"));
    steps.def(
        "convert_layout",
        [](const std::shared_ptr<ov::preprocess::PreProcessSteps>& me, const ov::Layout& layout = {}) {
            me->convert_layout(layout);
            return me;
        },
        py::arg("dst_layout"));
    steps.def(
        "convert_layout",
        [](const std::shared_ptr<ov::preprocess::PreProcessSteps>& me, const std::vector<uint64_t>& dims) {
            me->convert_layout(dims);
            return me;
        },
        py::arg("dims"));
    steps.def("reverse_channels", [](const std::shared_ptr<ov::preprocess::PreProcessSteps>& me) {
        me->reverse_channels();
        return me;
    });
}

static void regclass_graph_PostProcessSteps(py::module m) {
    py::class_<ov::preprocess::PostProcessSteps, std::shared_ptr<ov::preprocess::PostProcessSteps>> steps(
        m,
        "PostProcessSteps");
    steps.doc() = "openvino.impl.preprocess.PostprocessSteps wraps ov::preprocess::PostProcessSteps";

    steps.def(py::init<>());

    steps.def(
        "convert_element_type",
        [](const std::shared_ptr<ov::preprocess::PostProcessSteps>& me, ov::element::Type type) {
            me->convert_element_type(type);
            return me;
        },
        py::arg("type"),
        R"(
                Converts tensor element type to specified type.
                Tensor must have openvino.Type.f32 data type.
                Parameters
                ----------
                type : Type
                    Destination type.
                Returns
                ----------
                convert_element_type : PostProcessSteps
                    Reference to itself to allow chaining of calls in client's code in a builder-like manner.
              )");
    steps.def(
        "convert_layout",
        [](const std::shared_ptr<ov::preprocess::PostProcessSteps>& me, const ov::Layout& layout = {}) {
            me->convert_layout(layout);
            return me;
        },
        py::arg("dst_layout"));
    steps.def(
        "convert_layout",
        [](const std::shared_ptr<ov::preprocess::PostProcessSteps>& me, const std::vector<uint64_t>& dims) {
            me->convert_layout(dims);
            return me;
        },
        py::arg("dims"));
    steps.def(
        "custom",
        [](const std::shared_ptr<ov::preprocess::PostProcessSteps>& me, py::function op) {
            me->custom(op.cast<const ov::preprocess::PostProcessSteps::CustomPostprocessOp>());
            return me;
        },
        py::arg("operation"),
        R"(
                Adds custom postprocessing operation.
                Parameters
                ----------
                operation : function taking Output<Node> as input argument and returning Output<Node> after postprocessing.
                Returns
                ----------
                custom : PostProcessSteps
                    Reference to itself to allow chaining of calls in client's code in a builder-like manner.
              )");
}

static void regclass_graph_InputTensorInfo(py::module m) {
    py::class_<ov::preprocess::InputTensorInfo, std::shared_ptr<ov::preprocess::InputTensorInfo>> info(
        m,
        "InputTensorInfo");
    info.doc() = "openvino.impl.preprocess.InputTensorInfo wraps ov::preprocess::InputTensorInfo";

    info.def(py::init<>());

    info.def(
        "set_element_type",
        [](const std::shared_ptr<ov::preprocess::InputTensorInfo>& me, const ov::element::Type& type) {
            me->set_element_type(type);
            return me;
        },
        py::arg("type"),
        R"(
                Set initial client's tensor element type. If type is not the same as network's element type,
                conversion of element type will be done automatically.
                Parameters
                ----------
                type : Type
                    Client's input tensor element type.
                Returns
                ----------
                tensor : InputTensorInfo
                    Reference to itself to allow chaining of calls in client's code in a builder-like manner.
              )");
    info.def("set_layout", [](const std::shared_ptr<ov::preprocess::InputTensorInfo>& me, const ov::Layout& layout) {
        me->set_layout(layout);
        return me;
    });
    info.def("set_spatial_dynamic_shape", [](const std::shared_ptr<ov::preprocess::InputTensorInfo>& me) {
        me->set_spatial_dynamic_shape();
        return me;
    });
    info.def("set_spatial_static_shape",
             [](const std::shared_ptr<ov::preprocess::InputTensorInfo>& me, size_t height, size_t width) {
                 me->set_spatial_static_shape(height, width);
                 return me;
             });
    info.def("set_color_format",
             [](const std::shared_ptr<ov::preprocess::InputTensorInfo>& me,
                const ov::preprocess::ColorFormat& format,
                const std::vector<std::string>& sub_names = {}) {
                 me->set_color_format(format, sub_names);
                 return me;
             });
}

static void regclass_graph_OutputTensorInfo(py::module m) {
    py::class_<ov::preprocess::OutputTensorInfo, std::shared_ptr<ov::preprocess::OutputTensorInfo>> info(
        m,
        "OutputTensorInfo");
    info.doc() = "openvino.impl.preprocess.OutputTensorInfo wraps ov::preprocess::OutputTensorInfo";

    info.def(py::init<>());

    info.def(
        "set_element_type",
        [](const std::shared_ptr<ov::preprocess::OutputTensorInfo>& me, const ov::element::Type& type) {
            me->set_element_type(type);
            return me;
        },
        py::arg("type"),
        R"(
                Set client's output tensor element type. If type is not the same as network's element type,
                conversion of element type will be done automatically.
                Parameters
                ----------
                type : Type
                    Client's output tensor element type.
                Returns
                ----------
                tensor : OutputTensorInfo
                    Reference to itself to allow chaining of calls in client's code in a builder-like manner.
              )");
    info.def("set_layout", [](const std::shared_ptr<ov::preprocess::OutputTensorInfo>& me, const ov::Layout& layout) {
        me->set_layout(layout);
        return me;
    });
}

static void regclass_graph_InputInfo(py::module m) {
    py::class_<ov::preprocess::InputInfo, std::shared_ptr<ov::preprocess::InputInfo>> inp(m, "InputInfo");
    inp.doc() = "openvino.impl.preprocess.InputInfo wraps ov::preprocess::InputInfo";

    inp.def(py::init<>(), R"(Default constructor, can be used only for networks with exactly one input)");
    inp.def(py::init<size_t>(), R"(Constructor with parameter index as argument)");
    inp.def(py::init<const std::string&>(), R"(Constructor with input tensor name as argument)");

    inp.def(
        "tensor",
        [](const std::shared_ptr<ov::preprocess::InputInfo>& me,
           const std::shared_ptr<ov::preprocess::InputTensorInfo>& inputTensorInfo) {
            me->tensor(std::move(*inputTensorInfo));
            return me;
        },
        py::arg("tensor"),
        R"(
                Adds builder for actual tensor information of client's input.
                Parameters
                ----------
                tensor : InputTensorInfo
                    Client's input tensor information. It's internal data will be moved to parent InputInfo object.
                Returns
                ----------
                tensor : InputInfo
                    Reference to itself to allow chaining of calls in client's code in a builder-like manner.
              )");
    inp.def(
        "preprocess",
        [](const std::shared_ptr<ov::preprocess::InputInfo>& me,
           const std::shared_ptr<ov::preprocess::PreProcessSteps>& preProcessSteps) {
            me->preprocess(std::move(*preProcessSteps));
            return me;
        },
        py::arg("pre_process_steps"),
        R"(
                Adds builder for actual preprocessing steps for input parameter.
                Steps can specify various actions, like 'mean', 'scale' and others.
                Parameters
                ----------
                pre_process_steps : PreProcessSteps
                    Preprocessing steps. It's internal data will be moved to parent InputInfo object.
                Returns
                ----------
                preprocess : InputInfo
                    Reference to itself to allow chaining of calls in client's code in a builder-like manner.
              )");
    inp.def(
        "network",
        [](const std::shared_ptr<ov::preprocess::InputInfo>& me,
           const std::shared_ptr<ov::preprocess::InputNetworkInfo>& inputNetworkInfo) {
            me->network(std::move(*inputNetworkInfo));
            return me;
        },
        py::arg("input_network_info"));
}

static void regclass_graph_OutputInfo(py::module m) {
    py::class_<ov::preprocess::OutputInfo, std::shared_ptr<ov::preprocess::OutputInfo>> out(m, "OutputInfo");
    out.doc() = "openvino.impl.preprocess.OutputInfo wraps ov::preprocess::OutputInfo";

    out.def(py::init<>(), R"(Default constructor, can be used only for networks with exactly one output)");
    out.def(py::init<size_t>(), R"(Constructor with parameter index as argument)");
    out.def(py::init<const std::string&>(), R"(Constructor with tensor name as argument)");

    out.def(
        "tensor",
        [](const std::shared_ptr<ov::preprocess::OutputInfo>& me,
           const std::shared_ptr<ov::preprocess::OutputTensorInfo>& outputTensorInfo) {
            me->tensor(std::move(*outputTensorInfo));
            return me;
        },
        py::arg("tensor"),
        R"(
                Adds builder for actual tensor information of client's output.
                Parameters
                ----------
                tensor : OutputTensorInfo
                    Client's output tensor information. It's internal data will be moved to parent OutputInfo object.
                Returns
                ----------
                tensor : OutputInfo
                    Reference to itself to allow chaining of calls in client's code in a builder-like manner.
              )");
    out.def(
        "postprocess",
        [](const std::shared_ptr<ov::preprocess::OutputInfo>& me,
           const std::shared_ptr<ov::preprocess::PostProcessSteps>& postProcessSteps) {
            me->postprocess(std::move(*postProcessSteps));
            return me;
        },
        py::arg("post_process_steps"),
        R"(
                Adds builder for actual postprocessing steps for output parameter.
                Parameters
                ----------
                post_process_steps : PostProcessSteps
                    Postprocessing steps. It's internal data will be moved to parent OutputInfo object.
                Returns
                ----------
                preprocess : OutputInfo
                    Reference to itself to allow chaining of calls in client's code in a builder-like manner.
              )");
    out.def(
        "network",
        [](const std::shared_ptr<ov::preprocess::OutputInfo>& me,
           const std::shared_ptr<ov::preprocess::OutputNetworkInfo>& outputNetworkInfo) {
            me->network(std::move(*outputNetworkInfo));
            return me;
        },
        py::arg("output_network_info"));
}

static void regclass_graph_OutputNetworkInfo(py::module m) {
    py::class_<ov::preprocess::OutputNetworkInfo, std::shared_ptr<ov::preprocess::OutputNetworkInfo>> info(
        m,
        "OutputNetworkInfo");
    info.doc() = "openvino.impl.preprocess.OutputNetworkInfo wraps ov::preprocess::OutputNetworkInfo";

    info.def(py::init<>());

    info.def("set_layout", [](const std::shared_ptr<ov::preprocess::OutputNetworkInfo>& me, const ov::Layout& layout) {
        me->set_layout(layout);
        return me;
    });
}

static void regclass_graph_InputNetworkInfo(py::module m) {
    py::class_<ov::preprocess::InputNetworkInfo, std::shared_ptr<ov::preprocess::InputNetworkInfo>> info(
        m,
        "InputNetworkInfo");
    info.doc() = "openvino.impl.preprocess.InputNetworkInfo wraps ov::preprocess::InputNetworkInfo";

    info.def(py::init<>());

    info.def("set_layout", [](const std::shared_ptr<ov::preprocess::InputNetworkInfo>& me, const ov::Layout& layout) {
        me->set_layout(layout);
        return me;
    });
}

static void regenum_graph_ColorFormat(py::module m) {
    py::enum_<ov::preprocess::ColorFormat>(m, "ColorFormat")
        .value("UNDEFINED", ov::preprocess::ColorFormat::UNDEFINED)
        .value("NV12_SINGLE_PLANE", ov::preprocess::ColorFormat::NV12_SINGLE_PLANE)
        .value("NV12_TWO_PLANES", ov::preprocess::ColorFormat::NV12_TWO_PLANES)
        .value("RGB", ov::preprocess::ColorFormat::RGB)
        .value("BGR", ov::preprocess::ColorFormat::BGR)
        .export_values();
}

static void regenum_graph_ResizeAlgorithm(py::module m) {
    py::enum_<ov::preprocess::ResizeAlgorithm>(m, "ResizeAlgorithm")
        .value("RESIZE_LINEAR", ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR)
        .value("RESIZE_CUBIC", ov::preprocess::ResizeAlgorithm::RESIZE_CUBIC)
        .value("RESIZE_NEAREST", ov::preprocess::ResizeAlgorithm::RESIZE_NEAREST)
        .export_values();
}

void regclass_graph_PrePostProcessor(py::module m) {
    regclass_graph_PreProcessSteps(m);
    regclass_graph_PostProcessSteps(m);
    regclass_graph_InputInfo(m);
    regclass_graph_OutputInfo(m);
    regclass_graph_InputTensorInfo(m);
    regclass_graph_OutputTensorInfo(m);
    regclass_graph_InputNetworkInfo(m);
    regclass_graph_OutputNetworkInfo(m);
    regenum_graph_ColorFormat(m);
    regenum_graph_ResizeAlgorithm(m);
    py::class_<ov::preprocess::PrePostProcessor, std::shared_ptr<ov::preprocess::PrePostProcessor>> proc(
        m,
        "PrePostProcessor");
    proc.doc() = "openvino.impl.preprocess.PrePostProcessor wraps ov::preprocess::PrePostProcessor";

    proc.def(py::init<const std::shared_ptr<ov::Function>&>());

    proc.def(
        "input",
        [](const std::shared_ptr<ov::preprocess::PrePostProcessor>& me,
           const std::shared_ptr<ov::preprocess::InputInfo>& info) {
            me->input(std::move(*info));
            return me;
        },
        py::arg("input_info"),
        R"(
                Adds builder for preprocessing info for input parameter.
                Parameters
                ----------
                input_info : InputInfo
                    Preprocessing info for input parameter. It's internal data will be moved to PreProcessing object.
                Returns
                ----------
                in : PrePostProcessor
                    Reference to itself to allow chaining of calls in client's code.
              )");
    proc.def(
        "output",
        [](const std::shared_ptr<ov::preprocess::PrePostProcessor>& me,
           const std::shared_ptr<ov::preprocess::OutputInfo>& info) {
            me->output(std::move(*info));
            return me;
        },
        py::arg("output_info"),
        R"(
                Adds builder for preprocessing info for output parameter.
                Parameters
                ----------
                output_info : OutputInfo
                    Preprocessing info for output parameter. It's internal data will be moved to PreProcessing object.
                Returns
                ----------
                in : PrePostProcessor
                    Reference to itself to allow chaining of calls in client's code.
              )");
    proc.def("build", &ov::preprocess::PrePostProcessor::build);
}
