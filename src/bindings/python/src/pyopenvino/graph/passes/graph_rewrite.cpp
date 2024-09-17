// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/graph/passes/graph_rewrite.hpp"

#include <pybind11/pybind11.h>

#include <openvino/pass/backward_graph_rewrite.hpp>

#include "pyopenvino/core/common.hpp"

namespace py = pybind11;

void regclass_passes_GraphRewrite(py::module m) {
    py::class_<ov::pass::GraphRewrite, std::shared_ptr<ov::pass::GraphRewrite>, ov::pass::ModelPass, ov::pass::PassBase>
        graph_rewrite(m, "GraphRewrite");
    graph_rewrite.doc() =
        "openvino.runtime.passes.GraphRewrite executes sequence of MatcherPass transformations in topological order";

    graph_rewrite.def(py::init<>());
    graph_rewrite.def(py::init([](const std::shared_ptr<ov::pass::MatcherPass>& pass) {
                          return std::make_shared<ov::pass::GraphRewrite>(pass);
                      }),
                      py::arg("pass"),
                      R"(
                      Register single MatcherPass pass inside GraphRewrite.

                      :param pass: openvino.runtime.passes.MatcherPass instance
                      :type pass: openvino.runtime.passes.MatcherPass
    )");

    graph_rewrite.def("add_matcher",
                      static_cast<std::shared_ptr<ov::pass::MatcherPass> (ov::pass::GraphRewrite::*)(
                          const std::shared_ptr<ov::pass::MatcherPass>&)>(&ov::pass::GraphRewrite::add_matcher),
                      py::arg("pass"),
                      R"(
                      Register single MatcherPass pass inside GraphRewrite.

                      :param pass: openvino.runtime.passes.MatcherPass instance
                      :type pass: openvino.runtime.passes.MatcherPass
    )");

    py::class_<ov::pass::BackwardGraphRewrite,
               std::shared_ptr<ov::pass::BackwardGraphRewrite>,
               ov::pass::GraphRewrite,
               ov::pass::ModelPass,
               ov::pass::PassBase>
        back_graph_rewrite(m, "BackwardGraphRewrite");
    back_graph_rewrite.doc() = "openvino.runtime.passes.BackwardGraphRewrite executes sequence of MatcherPass "
                               "transformations in reversed topological order";

    back_graph_rewrite.def(py::init<>());
    back_graph_rewrite.def(py::init([](const std::shared_ptr<ov::pass::MatcherPass>& pass) {
                               return std::make_shared<ov::pass::BackwardGraphRewrite>(pass);
                           }),
                           py::arg("pass"),
                           R"(
                           Register single MatcherPass pass inside BackwardGraphRewrite.

                           :param pass: openvino.runtime.passes.MatcherPass instance
                           :type pass: openvino.runtime.passes.MatcherPass
    )");

    back_graph_rewrite.def(
        "add_matcher",
        static_cast<std::shared_ptr<ov::pass::MatcherPass> (ov::pass::BackwardGraphRewrite::*)(
            const std::shared_ptr<ov::pass::MatcherPass>&)>(&ov::pass::BackwardGraphRewrite::add_matcher),
        py::arg("pass"),
        R"(
        Register single MatcherPass pass inside BackwardGraphRewrite.

        :param pass: openvino.runtime.passes.MatcherPass instance
        :type pass: openvino.runtime.passes.MatcherPass
    )");

    back_graph_rewrite.def("__repr__", [](const ov::pass::BackwardGraphRewrite& self) {
        return Common::get_simple_repr(self);
    });
}
