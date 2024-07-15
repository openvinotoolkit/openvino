// Copyright (C) 2018-2024 Intel Corporation
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
        [](ov::preprocess::PreProcessSteps& self, float value) {
            return &self.mean(value);
        },
        py::arg("value"),
        R"(
            Subtracts single float value from each element in input tensor.
            Input tensor must have ov.Type.f32 data type.

            :param value: Value to subtract.
            :type value: float
            :return: Reference to itself to allow chaining of calls in client's code in a builder-like manner.
            :rtype: openvino.runtime.preprocess.PreProcessSteps
        )");

    steps.def(
        "mean",
        [](ov::preprocess::PreProcessSteps& self, const std::vector<float>& values) {
            return &self.mean(values);
        },
        py::arg("values"),
        R"(
            Subtracts a given single float value from each element in a given channel from input tensor.
            Input tensor must have ov.Type.f32 data type.

            :param values: Values to subtract.
            :type values: List[float]
            :return: Reference to itself to allow chaining of calls in client's code in a builder-like manner.
            :rtype: openvino.runtime.preprocess.PreProcessSteps
        )");

    steps.def(
        "scale",
        [](ov::preprocess::PreProcessSteps& self, float value) {
            return &self.scale(value);
        },
        py::arg("value"),
        R"(
            Divides each element in input tensor by specified constant float value.
            Input tensor must have ov.Type.f32 data type.

            :param value: Value used in division.
            :type value: float
            :return: Reference to itself to allow chaining of calls in client's code in a builder-like manner.
            :rtype: openvino.runtime.preprocess.PreProcessSteps
        )");

    steps.def(
        "scale",
        [](ov::preprocess::PreProcessSteps& self, const std::vector<float>& values) {
            return &self.scale(values);
        },
        py::arg("values"),
        R"(
            Divides each element in a given channel from input tensor by a given single float value.
            Input tensor must have ov.Type.f32 data type.

            :param values: Values which are used in division.
            :type values: List[float]
            :return: Reference to itself to allow chaining of calls in client's code in a builder-like manner.
            :rtype: openvino.runtime.preprocess.PreProcessSteps
        )");

    steps.def(
        "convert_element_type",
        [](ov::preprocess::PreProcessSteps& self, ov::element::Type type = {}) {
            return &self.convert_element_type(type);
        },
        py::arg_v("type", ov::element::undefined, "openvino.runtime.Type.undefined"),
        R"(
            Converts input tensor element type to specified type.
            Input tensor must have openvino.Type data type.

            :param type: Destination type. If not specified, type will be taken from model input's element type
            :type type: openvino.runtime.Type
            :return: Reference to itself to allow chaining of calls in client's code in a builder-like manner.
            :rtype: openvino.runtime.preprocess.PreProcessSteps
        )");

    steps.def(
        "custom",
        [](ov::preprocess::PreProcessSteps& self, py::function op) {
            return &self.custom(op.cast<const ov::preprocess::PreProcessSteps::CustomPreprocessOp>());
        },
        py::arg("operation"),
        R"(
            Adds custom preprocessing operation.

            :param operation: Python's function which takes `openvino.runtime.Output` as input argument and returns`openvino.runtime.Output`.
            :type operation: function
            :return: Reference to itself, allows chaining of calls in client's code in a builder-like manner.
            :rtype: openvino.runtime.preprocess.PreProcessSteps
        )");

    steps.def(
        "convert_color",
        [](ov::preprocess::PreProcessSteps& self, const ov::preprocess::ColorFormat& dst_format) {
            return &self.convert_color(dst_format);
        },
        py::arg("dst_format"));

    steps.def(
        "resize",
        [](ov::preprocess::PreProcessSteps& self,
           ov::preprocess::ResizeAlgorithm alg,
           size_t dst_height,
           size_t dst_width) {
            return &self.resize(alg, dst_height, dst_width);
        },
        py::arg("alg"),
        py::arg("dst_height"),
        py::arg("dst_width"));

    steps.def(
        "resize",
        [](ov::preprocess::PreProcessSteps& self, ov::preprocess::ResizeAlgorithm alg) {
            return &self.resize(alg);
        },
        py::arg("alg"));

    steps.def(
        "crop",
        [](ov::preprocess::PreProcessSteps& self, const std::vector<int>& begin, const std::vector<int>& end) {
            return &self.crop(begin, end);
        },
        py::arg("begin"),
        py::arg("end"));

    steps.def(
        "convert_layout",
        [](ov::preprocess::PreProcessSteps& self, const ov::Layout& layout = {}) {
            return &self.convert_layout(layout);
        },
        py::arg("dst_layout"));

    steps.def(
        "convert_layout",
        [](ov::preprocess::PreProcessSteps& self, const std::vector<uint64_t>& dims) {
            return &self.convert_layout(dims);
        },
        py::arg("dims"));

    steps.def("reverse_channels", [](ov::preprocess::PreProcessSteps& self) {
        return &self.reverse_channels();
    });

    steps.def(
        "pad",
        [](ov::preprocess::PreProcessSteps& self,
           const std::vector<int>& pads_begin,
           const std::vector<int>& pads_end,
           float value,
           ov::preprocess::PaddingMode mode) {
            return &self.pad(pads_begin, pads_end, value, mode);
        },
        py::arg("pads_begin"),
        py::arg("pads_end"),
        py::arg("value"),
        py::arg("mode"),
        R"(
            Adds padding preprocessing operation.

            :param pads_begin: Number of elements matches the number of indices in data attribute. Specifies the number of padding elements at the ending of each axis.
            :type pads_begin: 1D tensor of type T_INT.
            :param pads_end: Number of elements matches the number of indices in data attribute. Specifies the number of padding elements at the ending of each axis.
            :type pads_end: 1D tensor of type T_INT.
            :param value: All new elements are populated with this value or with 0 if input not provided. Shouldn’t be set for other pad_mode values.
            :type value: scalar tensor of type T. 
            :param mode: pad_mode specifies the method used to generate new element values.
            :type mode: string
            :return: Reference to itself, allows chaining of calls in client's code in a builder-like manner.
            :rtype: openvino.preprocess.PreProcessSteps
        )");

    steps.def(
        "pad",
        [](ov::preprocess::PreProcessSteps& self,
           const std::vector<int>& pads_begin,
           const std::vector<int>& pads_end,
           const std::vector<float>& values,
           ov::preprocess::PaddingMode mode) {
            return &self.pad(pads_begin, pads_end, values, mode);
        },
        py::arg("pads_begin"),
        py::arg("pads_end"),
        py::arg("value"),
        py::arg("mode"),
        R"(
            Adds padding preprocessing operation.

            :param pads_begin: Number of elements matches the number of indices in data attribute. Specifies the number of padding elements at the ending of each axis.
            :type pads_begin: 1D tensor of type T_INT.
            :param pads_end: Number of elements matches the number of indices in data attribute. Specifies the number of padding elements at the ending of each axis.
            :type pads_end: 1D tensor of type T_INT.
            :param value: All new elements are populated with this value or with 0 if input not provided. Shouldn’t be set for other pad_mode values.
            :type value: scalar tensor of type T. 
            :param mode: pad_mode specifies the method used to generate new element values.
            :type mode: string
            :return: Reference to itself, allows chaining of calls in client's code in a builder-like manner.
            :rtype: openvino.runtime.PreProcessSteps
        )");
}

static void regclass_graph_PostProcessSteps(py::module m) {
    py::class_<ov::preprocess::PostProcessSteps, Common::ref_wrapper<ov::preprocess::PostProcessSteps>> steps(
        m,
        "PostProcessSteps");
    steps.doc() = "openvino.runtime.preprocess.PostprocessSteps wraps ov::preprocess::PostProcessSteps";

    steps.def(
        "convert_element_type",
        [](ov::preprocess::PostProcessSteps& self, ov::element::Type type = {}) {
            return &self.convert_element_type(type);
        },
        py::arg_v("type", ov::element::undefined, "openvino.runtime.Type.undefined"),
        R"(
            Converts tensor element type to specified type.
            Tensor must have openvino.Type data type.

            :param type: Destination type. If not specified, type will be taken from model output's element type.
            :type type: openvino.runtime.Type
            :return: Reference to itself to allow chaining of calls in client's code in a builder-like manner.
            :rtype: openvino.runtime.preprocess.PostProcessSteps
        )");

    steps.def(
        "convert_layout",
        [](ov::preprocess::PostProcessSteps& self, const ov::Layout& layout = {}) {
            return &self.convert_layout(layout);
        },
        py::arg("dst_layout"));

    steps.def(
        "convert_layout",
        [](ov::preprocess::PostProcessSteps& self, const std::vector<uint64_t>& dims) {
            return &self.convert_layout(dims);
        },
        py::arg("dims"));

    steps.def(
        "custom",
        [](ov::preprocess::PostProcessSteps& self, py::function op) {
            return &self.custom(op.cast<const ov::preprocess::PostProcessSteps::CustomPostprocessOp>());
        },
        py::arg("operation"),
        R"(
            Adds custom postprocessing operation.

            :param operation: Python's function which takes `openvino.runtime.Output` as input argument and returns`openvino.runtime.Output`.
            :type operation: function
            :return: Reference to itself, allows chaining of calls in client's code in a builder-like manner.
            :rtype: openvino.runtime.preprocess.PreProcessSteps
        )");
}

static void regclass_graph_InputTensorInfo(py::module m) {
    py::class_<ov::preprocess::InputTensorInfo, Common::ref_wrapper<ov::preprocess::InputTensorInfo>> info(
        m,
        "InputTensorInfo");
    info.doc() = "openvino.runtime.preprocess.InputTensorInfo wraps ov::preprocess::InputTensorInfo";

    info.def(
        "set_element_type",
        [](ov::preprocess::InputTensorInfo& self, const ov::element::Type& type) {
            return &self.set_element_type(type);
        },
        py::arg("type"),
        R"(
            Set initial client's tensor element type. If type is not the same as model's element type,
            conversion of element type will be done automatically.

            :param type: Client's input tensor element type.
            :type type: openvino.runtime.Type
            :return: Reference to itself, allows chaining of calls in client's code in a builder-like manner.
            :rtype: openvino.runtime.preprocess.InputTensorInfo
        )");

    info.def(
        "set_layout",
        [](ov::preprocess::InputTensorInfo& self, const ov::Layout& layout) {
            return &self.set_layout(layout);
        },
        py::arg("layout"),
        R"(
            Set layout for input tensor info 
            :param layout: layout to be set
            :type layout: Union[str, openvino.runtime.Layout]
        )");

    info.def("set_spatial_dynamic_shape", [](ov::preprocess::InputTensorInfo& self) {
        return &self.set_spatial_dynamic_shape();
    });

    info.def(
        "set_spatial_static_shape",
        [](ov::preprocess::InputTensorInfo& self, size_t height, size_t width) {
            return &self.set_spatial_static_shape(height, width);
        },
        py::arg("height"),
        py::arg("width"));

    info.def(
        "set_shape",
        [](ov::preprocess::InputTensorInfo& self, const ov::PartialShape& shape) {
            return &self.set_shape(shape);
        },
        py::arg("shape"));

    // Allow to use set_shape([1,2,3]) in Python code, not set_shape(PartialShape([1,2,3]))
    info.def(
        "set_shape",
        [](ov::preprocess::InputTensorInfo& self, const std::vector<int64_t>& shape) {
            return &self.set_shape(shape);
        },
        py::arg("shape"));

    info.def(
        "set_color_format",
        [](ov::preprocess::InputTensorInfo& self,
           const ov::preprocess::ColorFormat& format,
           const std::vector<std::string>& sub_names = {}) {
            return &self.set_color_format(format, sub_names);
        },
        py::arg("format"),
        py::arg("sub_names") = std::vector<std::string>{});

    info.def(
        "set_memory_type",
        [](ov::preprocess::InputTensorInfo& self, const std::string& memory_type) {
            return &self.set_memory_type(memory_type);
        },
        py::arg("memory_type"));

    info.def(
        "set_from",
        [](ov::preprocess::InputTensorInfo& self, const ov::Tensor& tensor) {
            return &self.set_from(tensor);
        },
        py::arg("runtime_tensor"),
        R"(
            Helper function to reuse element type and shape from user's created tensor. Overwrites previously
            set shape and element type via `set_shape` and `set_element_type' methods. This method should be
            used only in case if runtime tensor is already known and avaiable before.

            :param runtime_tensor: User's created tensor
            :type type: openvino.runtime.Tensor
            :return: Reference to itself, allows chaining of calls in client's code in a builder-like manner.
            :rtype: openvino.runtime.preprocess.InputTensorInfo
        )");

    info.def(
        "set_from",
        [](ov::preprocess::InputTensorInfo& self, py::array& numpy_array) {
            // Convert to contiguous array if not already C-style.
            return &self.set_from(Common::object_from_data<ov::Tensor>(numpy_array, false));
        },
        py::arg("runtime_tensor"),
        R"(
            Helper function to reuse element type and shape from user's created tensor. Overwrites previously
            set shape and element type via `set_shape` and `set_element_type' methods. This method should be
            used only in case if runtime tensor is already known and avaiable before.

            :param runtime_tensor: User's created numpy array
            :type type: numpy.ndarray
            :return: Reference to itself, allows chaining of calls in client's code in a builder-like manner.
            :rtype: openvino.runtime.preprocess.InputTensorInfo
        )");
}

static void regclass_graph_OutputTensorInfo(py::module m) {
    py::class_<ov::preprocess::OutputTensorInfo, Common::ref_wrapper<ov::preprocess::OutputTensorInfo>> info(
        m,
        "OutputTensorInfo");
    info.doc() = "openvino.runtime.preprocess.OutputTensorInfo wraps ov::preprocess::OutputTensorInfo";

    info.def(
        "set_element_type",
        [](ov::preprocess::OutputTensorInfo& self, const ov::element::Type& type) {
            return &self.set_element_type(type);
        },
        py::arg("type"),
        R"(
            Set client's output tensor element type. If type is not the same as model's element type,
            conversion of element type will be done automatically.

            :param type: Client's output tensor element type.
            :type type: openvino.runtime.Type
            :return: Reference to itself to allow chaining of calls in client's code in a builder-like manner.
            :rtype: openvino.runtime.preprocess.OutputTensorInfo
        )");

    info.def(
        "set_layout",
        [](ov::preprocess::OutputTensorInfo& self, const ov::Layout& layout) {
            return &self.set_layout(layout);
        },
        py::arg("layout"),
        R"(
            Set layout for output tensor info 
            :param layout: layout to be set
            :type layout: Union[str, openvino.runtime.Layout]
        )");
}

static void regclass_graph_InputInfo(py::module m) {
    py::class_<ov::preprocess::InputInfo, Common::ref_wrapper<ov::preprocess::InputInfo>> inp(m, "InputInfo");
    inp.doc() = "openvino.runtime.preprocess.InputInfo wraps ov::preprocess::InputInfo";

    inp.def("tensor", [](ov::preprocess::InputInfo& self) {
        return &self.tensor();
    });

    inp.def("preprocess", [](ov::preprocess::InputInfo& self) {
        return &self.preprocess();
    });

    inp.def("model", [](ov::preprocess::InputInfo& self) {
        return &self.model();
    });
}

static void regclass_graph_OutputInfo(py::module m) {
    py::class_<ov::preprocess::OutputInfo, Common::ref_wrapper<ov::preprocess::OutputInfo>> out(m, "OutputInfo");
    out.doc() = "openvino.runtime.preprocess.OutputInfo wraps ov::preprocess::OutputInfo";

    out.def("tensor", [](ov::preprocess::OutputInfo& self) {
        return &self.tensor();
    });

    out.def("postprocess", [](ov::preprocess::OutputInfo& self) {
        return &self.postprocess();
    });

    out.def("model", [](ov::preprocess::OutputInfo& self) {
        return &self.model();
    });
}

static void regclass_graph_OutputModelInfo(py::module m) {
    py::class_<ov::preprocess::OutputModelInfo, Common::ref_wrapper<ov::preprocess::OutputModelInfo>> info(
        m,
        "OutputModelInfo");
    info.doc() = "openvino.runtime.preprocess.OutputModelInfo wraps ov::preprocess::OutputModelInfo";

    info.def(
        "set_layout",
        [](ov::preprocess::OutputModelInfo& self, const ov::Layout& layout) {
            return &self.set_layout(layout);
        },
        py::arg("layout"),
        R"(
            Set layout for output model info 
            :param layout: layout to be set
            :type layout: Union[str, openvino.runtime.Layout]
        )");
}

static void regclass_graph_InputModelInfo(py::module m) {
    py::class_<ov::preprocess::InputModelInfo, Common::ref_wrapper<ov::preprocess::InputModelInfo>> info(
        m,
        "InputModelInfo");
    info.doc() = "openvino.runtime.preprocess.InputModelInfo wraps ov::preprocess::InputModelInfo";

    info.def(
        "set_layout",
        [](ov::preprocess::InputModelInfo& self, const ov::Layout& layout) {
            return &self.set_layout(layout);
        },
        py::arg("layout"),
        R"(
            Set layout for input model
            :param layout: layout to be set
            :type layout: Union[str, openvino.runtime.Layout]
        )");
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
        .value("GRAY", ov::preprocess::ColorFormat::GRAY)
        .value("RGBX", ov::preprocess::ColorFormat::RGBX)
        .value("BGRX", ov::preprocess::ColorFormat::BGRX)
        .export_values();
}

static void regenum_graph_ResizeAlgorithm(py::module m) {
    py::enum_<ov::preprocess::ResizeAlgorithm>(m, "ResizeAlgorithm")
        .value("RESIZE_LINEAR", ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR)
        .value("RESIZE_CUBIC", ov::preprocess::ResizeAlgorithm::RESIZE_CUBIC)
        .value("RESIZE_NEAREST", ov::preprocess::ResizeAlgorithm::RESIZE_NEAREST)
        .value("RESIZE_BILINEAR_PILLOW", ov::preprocess::ResizeAlgorithm::RESIZE_BILINEAR_PILLOW)
        .value("RESIZE_BICUBIC_PILLOW", ov::preprocess::ResizeAlgorithm::RESIZE_BICUBIC_PILLOW)
        .export_values();
}

static void regenum_graph_PaddingMode(py::module m) {
    py::enum_<ov::preprocess::PaddingMode>(m, "PaddingMode")
        .value("CONSTANT", ov::preprocess::PaddingMode::CONSTANT)
        .value("REFLECT", ov::preprocess::PaddingMode::REFLECT)
        .value("SYMMETRIC", ov::preprocess::PaddingMode::SYMMETRIC)
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
    regenum_graph_PaddingMode(m);
    py::class_<ov::preprocess::PrePostProcessor, std::shared_ptr<ov::preprocess::PrePostProcessor>> proc(
        m,
        "PrePostProcessor");
    proc.doc() = "openvino.runtime.preprocess.PrePostProcessor wraps ov::preprocess::PrePostProcessor";

    proc.def(py::init<const std::shared_ptr<ov::Model>&>(), py::arg("model"));

    proc.def("input", [](ov::preprocess::PrePostProcessor& self) {
        return &self.input();
    });

    proc.def(
        "input",
        [](ov::preprocess::PrePostProcessor& self, const std::string& tensor_name) {
            return &self.input(tensor_name);
        },
        py::arg("tensor_name"));

    proc.def(
        "input",
        [](ov::preprocess::PrePostProcessor& self, size_t input_index) {
            return &self.input(input_index);
        },
        py::arg("input_index"));

    proc.def("output", [](ov::preprocess::PrePostProcessor& self) {
        return &self.output();
    });

    proc.def(
        "output",
        [](ov::preprocess::PrePostProcessor& self, const std::string& tensor_name) {
            return &self.output(tensor_name);
        },
        py::arg("tensor_name"));

    proc.def(
        "output",
        [](ov::preprocess::PrePostProcessor& self, size_t output_index) {
            return &self.output(output_index);
        },
        py::arg("output_index"));

    proc.def("build", &ov::preprocess::PrePostProcessor::build, py::call_guard<py::gil_scoped_release>());

    proc.def("__str__", [](const ov::preprocess::PrePostProcessor& self) -> std::string {
        std::stringstream ss;
        ss << self;
        return ss.str();
    });

    proc.def("__repr__", [](const ov::preprocess::PrePostProcessor& self) -> std::string {
        return "<" + Common::get_class_name(self) + ": " + py::cast(self).attr("__str__")().cast<std::string>() + ">";
    });
}
