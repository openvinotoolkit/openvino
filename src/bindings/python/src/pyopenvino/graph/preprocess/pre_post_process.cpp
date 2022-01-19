// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/graph/preprocess/pre_post_process.hpp"

#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include "openvino/core/model.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "pyopenvino/core/common.hpp"

namespace py = pybind11;

// Custom holder wrapping returned references to preprocessing objects
PYBIND11_DECLARE_HOLDER_TYPE(T, Common::ref_wrapper<T>)

static void regclass_graph_PreProcessSteps(py::module m) {
    py::class_<ov::preprocess::PreProcessSteps, Common::ref_wrapper<ov::preprocess::PreProcessSteps>> steps(
        m,
        "PreProcessSteps");
    steps.doc() = "openvino.runtime.preprocess.PreProcessSteps wraps ov::preprocess::PreProcessSteps";

    steps.def(
        "mean",
        [](ov::preprocess::PreProcessSteps& me, float value) {
            return &me.mean(value);
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
        [](ov::preprocess::PreProcessSteps& me, const std::vector<float>& values) {
            return &me.mean(values);
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
        [](ov::preprocess::PreProcessSteps& me, float value) {
            return &me.scale(value);
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
        [](ov::preprocess::PreProcessSteps& me, const std::vector<float>& values) {
            return &me.scale(values);
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
        [](ov::preprocess::PreProcessSteps& me, ov::element::Type type) {
            return &me.convert_element_type(type);
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
        [](ov::preprocess::PreProcessSteps& me, py::function op) {
            return &me.custom(op.cast<const ov::preprocess::PreProcessSteps::CustomPreprocessOp>());
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
        [](ov::preprocess::PreProcessSteps& me, const ov::preprocess::ColorFormat& dst_format) {
            return &me.convert_color(dst_format);
        },
        py::arg("dst_format"));
    steps.def(
        "resize",
        [](ov::preprocess::PreProcessSteps& me,
           ov::preprocess::ResizeAlgorithm alg,
           size_t dst_height,
           size_t dst_width) {
            return &me.resize(alg, dst_height, dst_width);
        },
        py::arg("alg"),
        py::arg("dst_height"),
        py::arg("dst_width"));
    steps.def(
        "resize",
        [](ov::preprocess::PreProcessSteps& me, ov::preprocess::ResizeAlgorithm alg) {
            return &me.resize(alg);
        },
        py::arg("alg"));
    steps.def(
        "convert_layout",
        [](ov::preprocess::PreProcessSteps& me, const ov::Layout& layout = {}) {
            return &me.convert_layout(layout);
        },
        py::arg("dst_layout"));
    steps.def(
        "convert_layout",
        [](ov::preprocess::PreProcessSteps& me, const std::vector<uint64_t>& dims) {
            return &me.convert_layout(dims);
        },
        py::arg("dims"));
    steps.def("reverse_channels", [](ov::preprocess::PreProcessSteps& me) {
        return &me.reverse_channels();
    });
}

static void regclass_graph_PostProcessSteps(py::module m) {
    py::class_<ov::preprocess::PostProcessSteps, Common::ref_wrapper<ov::preprocess::PostProcessSteps>> steps(
        m,
        "PostProcessSteps");
    steps.doc() = "openvino.runtime.preprocess.PostprocessSteps wraps ov::preprocess::PostProcessSteps";

    steps.def(
        "convert_element_type",
        [](ov::preprocess::PostProcessSteps& me, ov::element::Type type) {
            return &me.convert_element_type(type);
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
        [](ov::preprocess::PostProcessSteps& me, const ov::Layout& layout = {}) {
            return &me.convert_layout(layout);
        },
        py::arg("dst_layout"));
    steps.def(
        "convert_layout",
        [](ov::preprocess::PostProcessSteps& me, const std::vector<uint64_t>& dims) {
            return &me.convert_layout(dims);
        },
        py::arg("dims"));
    steps.def(
        "custom",
        [](ov::preprocess::PostProcessSteps& me, py::function op) {
            return &me.custom(op.cast<const ov::preprocess::PostProcessSteps::CustomPostprocessOp>());
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
    py::class_<ov::preprocess::InputTensorInfo, Common::ref_wrapper<ov::preprocess::InputTensorInfo>> info(
        m,
        "InputTensorInfo");
    info.doc() = "openvino.runtime.preprocess.InputTensorInfo wraps ov::preprocess::InputTensorInfo";

    info.def(
        "set_element_type",
        [](ov::preprocess::InputTensorInfo& me, const ov::element::Type& type) {
            return &me.set_element_type(type);
        },
        py::arg("type"),
        R"(
                Set initial client's tensor element type. If type is not the same as model's element type,
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
    info.def("set_layout", [](ov::preprocess::InputTensorInfo& me, const ov::Layout& layout) {
        return &me.set_layout(layout);
    });
    info.def("set_spatial_dynamic_shape", [](ov::preprocess::InputTensorInfo& me) {
        return &me.set_spatial_dynamic_shape();
    });
    info.def("set_spatial_static_shape", [](ov::preprocess::InputTensorInfo& me, size_t height, size_t width) {
        return &me.set_spatial_static_shape(height, width);
    });
    info.def("set_shape", [](ov::preprocess::InputTensorInfo& me, const ov::PartialShape& shape) {
        return &me.set_shape(shape);
    });
    // Allow to use set_shape([1,2,3]) in Python code, not set_shape(PartialShape([1,2,3]))
    info.def("set_shape", [](ov::preprocess::InputTensorInfo& me, const std::vector<int64_t>& shape) {
        return &me.set_shape(shape);
    });
    info.def("set_color_format",
             [](ov::preprocess::InputTensorInfo& me,
                const ov::preprocess::ColorFormat& format,
                const std::vector<std::string>& sub_names = {}) {
                 return &me.set_color_format(format, sub_names);
             });
    info.def("set_memory_type", [](ov::preprocess::InputTensorInfo& me, const std::string& memory_type) {
        return &me.set_memory_type(memory_type);
    });
}

static void regclass_graph_OutputTensorInfo(py::module m) {
    py::class_<ov::preprocess::OutputTensorInfo, Common::ref_wrapper<ov::preprocess::OutputTensorInfo>> info(
        m,
        "OutputTensorInfo");
    info.doc() = "openvino.runtime.preprocess.OutputTensorInfo wraps ov::preprocess::OutputTensorInfo";

    info.def(
        "set_element_type",
        [](ov::preprocess::OutputTensorInfo& me, const ov::element::Type& type) {
            return &me.set_element_type(type);
        },
        py::arg("type"),
        R"(
                Set client's output tensor element type. If type is not the same as model's element type,
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
    info.def("set_layout", [](ov::preprocess::OutputTensorInfo& me, const ov::Layout& layout) {
        return &me.set_layout(layout);
    });
}

static void regclass_graph_InputInfo(py::module m) {
    py::class_<ov::preprocess::InputInfo, Common::ref_wrapper<ov::preprocess::InputInfo>> inp(m, "InputInfo");
    inp.doc() = "openvino.runtime.preprocess.InputInfo wraps ov::preprocess::InputInfo";

    inp.def("tensor", [](ov::preprocess::InputInfo& me) {
        return &me.tensor();
    });
    inp.def("preprocess", [](ov::preprocess::InputInfo& me) {
        return &me.preprocess();
    });
    inp.def("model", [](ov::preprocess::InputInfo& me) {
        return &me.model();
    });
}

static void regclass_graph_OutputInfo(py::module m) {
    py::class_<ov::preprocess::OutputInfo, Common::ref_wrapper<ov::preprocess::OutputInfo>> out(m, "OutputInfo");
    out.doc() = "openvino.runtime.preprocess.OutputInfo wraps ov::preprocess::OutputInfo";

    out.def("tensor", [](ov::preprocess::OutputInfo& me) {
        return &me.tensor();
    });
    out.def("postprocess", [](ov::preprocess::OutputInfo& me) {
        return &me.postprocess();
    });
    out.def("model", [](ov::preprocess::OutputInfo& me) {
        return &me.model();
    });
}

static void regclass_graph_OutputModelInfo(py::module m) {
    py::class_<ov::preprocess::OutputModelInfo, Common::ref_wrapper<ov::preprocess::OutputModelInfo>> info(
        m,
        "OutputModelInfo");
    info.doc() = "openvino.runtime.preprocess.OutputModelInfo wraps ov::preprocess::OutputModelInfo";

    info.def("set_layout", [](ov::preprocess::OutputModelInfo& me, const ov::Layout& layout) {
        return &me.set_layout(layout);
    });
}

static void regclass_graph_InputModelInfo(py::module m) {
    py::class_<ov::preprocess::InputModelInfo, Common::ref_wrapper<ov::preprocess::InputModelInfo>> info(
        m,
        "InputModelInfo");
    info.doc() = "openvino.runtime.preprocess.InputModelInfo wraps ov::preprocess::InputModelInfo";

    info.def("set_layout", [](ov::preprocess::InputModelInfo& me, const ov::Layout& layout) {
        return &me.set_layout(layout);
    });
}

static void regenum_graph_ColorFormat(py::module m) {
    py::enum_<ov::preprocess::ColorFormat>(m, "ColorFormat")
        .value("UNDEFINED", ov::preprocess::ColorFormat::UNDEFINED)
        .value("NV12_SINGLE_PLANE", ov::preprocess::ColorFormat::NV12_SINGLE_PLANE)
        .value("NV12_TWO_PLANES", ov::preprocess::ColorFormat::NV12_TWO_PLANES)
        .value("I420_SINGLE_PLANE", ov::preprocess::ColorFormat::I420_SINGLE_PLANE)
        .value("I420_THREE_PLANES", ov::preprocess::ColorFormat::I420_THREE_PLANES)
        .value("RGB", ov::preprocess::ColorFormat::RGB)
        .value("BGR", ov::preprocess::ColorFormat::BGR)
        .value("RGBX", ov::preprocess::ColorFormat::RGBX)
        .value("BGRX", ov::preprocess::ColorFormat::BGRX)
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
    regclass_graph_InputModelInfo(m);
    regclass_graph_OutputModelInfo(m);
    regenum_graph_ColorFormat(m);
    regenum_graph_ResizeAlgorithm(m);
    py::class_<ov::preprocess::PrePostProcessor, std::shared_ptr<ov::preprocess::PrePostProcessor>> proc(
        m,
        "PrePostProcessor");
    proc.doc() = "openvino.runtime.preprocess.PrePostProcessor wraps ov::preprocess::PrePostProcessor";

    proc.def(py::init<const std::shared_ptr<ov::Model>&>());

    proc.def("input", [](ov::preprocess::PrePostProcessor& me) {
        return &me.input();
    });
    proc.def(
        "input",
        [](ov::preprocess::PrePostProcessor& me, const std::string& tensor_name) {
            return &me.input(tensor_name);
        },
        py::arg("tensor_name"));
    proc.def(
        "input",
        [](ov::preprocess::PrePostProcessor& me, size_t input_index) {
            return &me.input(input_index);
        },
        py::arg("input_index"));
    proc.def("output", [](ov::preprocess::PrePostProcessor& me) {
        return &me.output();
    });
    proc.def(
        "output",
        [](ov::preprocess::PrePostProcessor& me, const std::string& tensor_name) {
            return &me.output(tensor_name);
        },
        py::arg("tensor_name"));
    proc.def(
        "output",
        [](ov::preprocess::PrePostProcessor& me, size_t output_index) {
            return &me.output(output_index);
        },
        py::arg("output_index"));
    proc.def("build", &ov::preprocess::PrePostProcessor::build);

    proc.def("__str__", [](const ov::preprocess::PrePostProcessor& self) -> std::string {
        std::stringstream ss;
        ss << self;
        return ss.str();
    });

    proc.def("__repr__", [](const ov::preprocess::PrePostProcessor& self) -> std::string {
        return "<PrePostProcessor: " + py::cast(self).attr("__str__")().cast<std::string>() + ">";
    });
}
