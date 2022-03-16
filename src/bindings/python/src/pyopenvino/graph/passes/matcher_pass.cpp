// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include <string>

#include "pyopenvino/graph/passes/matcher_pass.hpp"

namespace py = pybind11;

void regclass_Matcher(py::module m) {
    py::class_<ov::pass::pattern::Matcher, std::shared_ptr<ov::pass::pattern::Matcher>> matcher(m, "Matcher");
    matcher.doc() = "openvino.runtime.passes.Matcher wraps ov::pass::pattern::Matcher";
    matcher.def(py::init([](const std::shared_ptr<ov::Node>& node,
                            const std::string& name) {
                    return std::make_shared<ov::pass::pattern::Matcher>(node, name);
                }),
                py::arg("node"),
                py::arg("name"));

    matcher.def(py::init([](ov::Output<ov::Node> & output,
                            const std::string& name) {
                    return std::make_shared<ov::pass::pattern::Matcher>(output, name);
                }),
                py::arg("output"),
                py::arg("name"));

    matcher.def("get_name", &ov::pass::pattern::Matcher::get_name);
    matcher.def("get_match_root", &ov::pass::pattern::Matcher::get_match_root);
    matcher.def("get_match_value", &ov::pass::pattern::Matcher::get_match_value);
    matcher.def("get_match_nodes", &ov::pass::pattern::Matcher::get_matched_nodes);
    matcher.def("get_match_values", static_cast<const ov::OutputVector& (ov::pass::pattern::Matcher::*)() const>(&ov::pass::pattern::Matcher::get_matched_values));
    matcher.def("get_pattern_value_map", &ov::pass::pattern::Matcher::get_pattern_value_map);

    matcher.def("match", static_cast<bool (ov::pass::pattern::Matcher::*)(const ov::Output<ov::Node>&)>(&ov::pass::pattern::Matcher::match));
    matcher.def("match", static_cast<bool (ov::pass::pattern::Matcher::*)(std::shared_ptr<ov::Node>)>(&ov::pass::pattern::Matcher::match));
}

class PyMatcherPass: public ov::pass::MatcherPass {
public:
    using ov::pass::MatcherPass::register_matcher;
};

void regclass_MatcherPass(py::module m) {
    py::class_<ov::pass::MatcherPass,
    std::shared_ptr<ov::pass::MatcherPass>,
    ov::pass::PassBase> matcher_pass(m, "MatcherPass");
    matcher_pass.doc() = "openvino.runtime.passes.MatcherPass wraps ov::pass::MatcherPass";
    matcher_pass.def(py::init<>());
    matcher_pass.def(py::init([](const std::shared_ptr<ov::pass::pattern::Matcher>& m,
                                 ov::matcher_pass_callback callback) {
                         return std::make_shared<ov::pass::MatcherPass>(m, callback);
                     }),
                     py::arg("m"),
                     py::arg("callback"));
    matcher_pass.def("apply", &ov::pass::MatcherPass::apply);
    matcher_pass.def("register_new_node", &ov::pass::MatcherPass::register_new_node_);
    matcher_pass.def("register_matcher", static_cast<void (ov::pass::MatcherPass::*)(const std::shared_ptr<ov::pass::pattern::Matcher>&,
                                                                                     const ov::graph_rewrite_callback& callback)>(&PyMatcherPass::register_matcher));
}
