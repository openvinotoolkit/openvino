// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/pass/pass.hpp>
#include <pybind11/pybind11.h>

#include <openvino/pass/graph_rewrite.hpp>

#include "pyopenvino/graph/passes/graph_rewrite.hpp"

namespace py = pybind11;

void regclass_GraphRewrite(py::module m) {
    py::class_<ov::pass::GraphRewrite,
               std::shared_ptr<ov::pass::GraphRewrite>,
               ov::pass::ModelPass,
               ov::pass::PassBase> graph_rewrite(m, "GraphRewrite");
    graph_rewrite.doc() = "openvino.runtime.passes.GraphRewrite execute sequence of MatcherPass transformations in topological order";

    graph_rewrite.def(py::init<>());
    graph_rewrite.def("add_matcher", static_cast<std::shared_ptr<ov::pass::MatcherPass> (ov::pass::GraphRewrite::*)(std::shared_ptr<ov::pass::MatcherPass>)>(&ov::pass::GraphRewrite::add_matcher));
}
