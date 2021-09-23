// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/preprocess/pre_post_process.hpp"

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "pyngraph/preprocess/pre_post_process.hpp"

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(const std::vector<float>);

static void regclass_pyngraph_PreProcessSteps(py::module m) {
    py::class_<ov::preprocess::PreProcessSteps, std::shared_ptr<ov::preprocess::PreProcessSteps>> steps(
        m,
        "PreProcessSteps");
    steps.doc() = "ngraph.impl.preprocess.PreProcessSteps wraps ov::preprocess::PreProcessSteps";

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
                Input tensor must have ngraph.Type.f32 data type.
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
        [](const std::shared_ptr<ov::preprocess::PreProcessSteps>& me, const std::vector<float> values) {
            me->mean(values);
            return me;
        },
        py::arg("values"),
        R"(
                Subtracts a given single float value from each element in a given channel from input tensor.
                Input tensor must have ngraph.Type.f32 data type.
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
                Input tensor must have ngraph.Type.f32 data type.
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
                Input tensor must have ngraph.Type.f32 data type.
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
                Input tensor must have ngraph.Type.f32 data type.
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
        [](const std::shared_ptr<ov::preprocess::PreProcessSteps>& me,
           const ov::preprocess::PreProcessSteps::CustomPreprocessOp& op) {
            me->custom(op);
            return me;
        },
        py::arg("operation"),
        R"(
                Adds custom preprocessing operation.
                Parameters
                ----------
                operation : function taking Node as input argument and returning Node after preprocessing.
                Returns
                ----------
                custom : PreProcessSteps
                    Reference to itself to allow chaining of calls in client's code in a builder-like manner.
              )");
}

static void regclass_pyngraph_InputTensorInfo(py::module m) {
    py::class_<ov::preprocess::InputTensorInfo, std::shared_ptr<ov::preprocess::InputTensorInfo>> info(
        m,
        "InputTensorInfo");
    info.doc() = "ngraph.impl.preprocess.InputTensorInfo wraps ov::preprocess::InputTensorInfo";

    info.def(py::init<>());
    info.def(
        "set_element_type",
        [](const std::shared_ptr<ov::preprocess::InputTensorInfo>& me, const ov::element::Type& type) {
            me->set_element_type(type);
            return me;
        },
        py::arg("type"),
        R"(
                Set initial client's tensor element type. If type is not the same as network's element type, user must
                add appropriate type conversion manually.
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
}

static void regclass_pyngraph_InputInfo(py::module m) {
    py::class_<ov::preprocess::InputInfo, std::shared_ptr<ov::preprocess::InputInfo>> inp(m, "InputInfo");
    inp.doc() = "ngraph.impl.preprocess.InputInfo wraps ov::preprocess::InputInfo";

    inp.def(py::init<>(), R"(Default constructor, can be used only for networks with exactly one input)");
    inp.def(py::init<size_t>(), R"(Constructor with parameter index as argument)");

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
}

void regclass_pyngraph_PrePostProcessor(py::module m) {
    regclass_pyngraph_PreProcessSteps(m);
    regclass_pyngraph_InputInfo(m);
    regclass_pyngraph_InputTensorInfo(m);
    py::class_<ov::preprocess::PrePostProcessor, std::shared_ptr<ov::preprocess::PrePostProcessor>> proc(
        m,
        "PrePostProcessor");
    proc.doc() = "ngraph.impl.preprocess.PrePostProcessor wraps ov::preprocess::PrePostProcessor";

    proc.def(py::init<>());
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
    proc.def("build",
             &ov::preprocess::PrePostProcessor::build,
             py::arg("function"),
             R"(
                Apply pre- and post-processing steps to specified model represented by `function` object.
                Parameters specified for inputs and outputs are validated on this stage
                and exception is raised if some data is invalid or inconsistent.
                Parameters
                ----------
                function : Function
                    Function representing existing model without pre-post-processing steps.
                Returns
                ----------
                build : Function
                    Same function object with applied pre(post)processing steps.
              )");
}
