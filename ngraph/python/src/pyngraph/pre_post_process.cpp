// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/pre_post_process/pre_post_process.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iterator>
#include <sstream>
#include <string>

#include "pyngraph/pre_post_process.hpp"

namespace py = pybind11;

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
                Subtracts single float value from each element in input tensor

                Parameters
                ----------
                value : float
                    Value to subtract

                Returns
                ----------
                mean : PreProcessSteps
                    Reference to itself to allow chaining of calls in client's code in a builder-like way
              )");
}

static void regclass_pyngraph_InputInfo(py::module m) {
    py::class_<ov::preprocess::InputInfo, std::shared_ptr<ov::preprocess::InputInfo>> inp(m, "InputInfo");
    inp.doc() = "ngraph.impl.preprocess.InputInfo wraps ov::preprocess::InputInfo";

    inp.def(py::init<>());
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
                Steps can specify various actions, like 'mean', 'scale', 'resize' and others

                Parameters
                ----------
                pre_process_steps : PreProcessSteps
                    Preprocessing steps. It's internal data will be moved to parent InputInfo object

                Returns
                ----------
                preprocess : InputInfo
                    Reference to itself to allow chaining of calls in client's code in a builder-like way
              )");
}

void regclass_pyngraph_PrePostProcessor(py::module m) {
    regclass_pyngraph_PreProcessSteps(m);
    regclass_pyngraph_InputInfo(m);
    py::class_<ov::preprocess::PrePostProcessor, std::shared_ptr<ov::preprocess::PrePostProcessor>> proc(
        m,
        "PrePostProcessor");
    proc.doc() = "ngraph.impl.preprocess.PrePostProcessor wraps ov::preprocess::PrePostProcessor";

    proc.def(py::init<>());
    proc.def(
        "input",
        [](const std::shared_ptr<ov::preprocess::PrePostProcessor>& me,
           const std::shared_ptr<ov::preprocess::InputInfo>& info) {
            me->in(std::move(*info));
            return me;
        },
        py::arg("input_info"),
        R"(
                Adds builder for preprocessing info for input parameter.

                Parameters
                ----------
                input_info : InputInfo
                    Preprocessing info for input parameter. 'input_info' internal data will be moved to PreProcessing object

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
                Parameters specified for inputs and outputs are validated on this stage and throw exception if some data is invalid

                Parameters
                ----------
                function : Function
                    Function representing existing model without pre-post-processing steps

                Returns
                ----------
                build : Function
                    Same function object with applied pre(post)processing steps
              )");
}
