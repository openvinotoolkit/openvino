// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/graph/passes/matcher_pass.hpp"

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"

namespace py = pybind11;

void regclass_passes_Matcher(py::module m) {
    py::class_<ov::pass::pattern::Matcher, std::shared_ptr<ov::pass::pattern::Matcher>> matcher(m, "Matcher");
    matcher.doc() = "openvino.runtime.passes.Matcher wraps ov::pass::pattern::Matcher";
    matcher.def(py::init([](const std::shared_ptr<ov::Node>& node, const std::string& name) {
                    return std::make_shared<ov::pass::pattern::Matcher>(node, name);
                }),
                py::arg("node"),
                py::arg("name"),
                R"(
                Creates Matcher object with given pattern root node and matcher name.
                Matcher object is used for pattern matching on Model.

                :param node: pattern root node.
                :type node: openvino.runtime.Node

                :param name: pattern name. Usually matches the MatcherPass class name.
                :type name: str
    )");

    matcher.def(py::init([](ov::Output<ov::Node>& output, const std::string& name) {
                    return std::make_shared<ov::pass::pattern::Matcher>(output, name);
                }),
                py::arg("output"),
                py::arg("name"),
                R"(
                Creates Matcher object with given pattern root node output and matcher name.
                Matcher object is used for pattern matching on Model.

                :param node: pattern root node output.
                :type node: openvino.runtime.Output

                :param name: pattern name. Usually matches the MatcherPass class name.
                :type name: str
    )");

    matcher.def("get_name",
                &ov::pass::pattern::Matcher::get_name,
                R"(
                Get Matcher name.

                :return: openvino.runtime.passes.Matcher name.
                :rtype: str
    )");

    matcher.def("get_match_root",
                &ov::pass::pattern::Matcher::get_match_root,
                R"(
                Get matched root node inside Model. Should be used after match() method is called.

                :return: matched node.
                :rtype: openvino.runtime.Node
    )");

    matcher.def("get_match_value",
                &ov::pass::pattern::Matcher::get_match_value,
                R"(
                Get matched node output inside Model. Should be used after match() method is called.

                :return: matched node output.
                :rtype: openvino.runtime.Output
    )");

    matcher.def("get_match_nodes",
                &ov::pass::pattern::Matcher::get_matched_nodes,
                R"(
                Get NodeVector of matched nodes. Should be used after match() method is called.

                :return: matched nodes vector.
                :rtype: List[openvino.runtime.Node]
    )");

    matcher.def("get_match_values",
                static_cast<const ov::OutputVector& (ov::pass::pattern::Matcher::*)() const>(
                    &ov::pass::pattern::Matcher::get_matched_values),
                R"(
                Get OutputVector of matched outputs. Should be used after match() method is called.

                :return: matched outputs vector.
                :rtype: List[openvino.runtime.Output]
    )");

    matcher.def("get_pattern_value_map",
                &ov::pass::pattern::Matcher::get_pattern_value_map,
                R"(
                Get map which can be used to access matched nodes using nodes from pattern.
                Should be used after match() method is called.

                :return: mapping of pattern nodes to matched nodes.
                :rtype: dict
    )");

    matcher.def("match",
                static_cast<bool (ov::pass::pattern::Matcher::*)(const ov::Output<ov::Node>&)>(
                    &ov::pass::pattern::Matcher::match),
                R"(
                Matches registered pattern starting from given output.

                :return: status of matching.
                :rtype: bool
    )");

    matcher.def("match",
                static_cast<bool (ov::pass::pattern::Matcher::*)(std::shared_ptr<ov::Node>)>(
                    &ov::pass::pattern::Matcher::match),
                R"(
                Matches registered pattern starting from given Node.

                :return: status of matching.
                :rtype: bool
    )");
}

class PyMatcherPass : public ov::pass::MatcherPass {
public:
    void py_register_matcher(const std::shared_ptr<ov::pass::pattern::Matcher>& matcher,
                             const ov::matcher_pass_callback& callback) {
        register_matcher(matcher, callback);
    }
};

void regclass_passes_MatcherPass(py::module m) {
    py::class_<ov::pass::MatcherPass, std::shared_ptr<ov::pass::MatcherPass>, ov::pass::PassBase, PyMatcherPass>
        matcher_pass(m, "MatcherPass");
    matcher_pass.doc() = "openvino.runtime.passes.MatcherPass wraps ov::pass::MatcherPass";
    matcher_pass.def(py::init<>());
    matcher_pass.def(
        py::init([](const std::shared_ptr<ov::pass::pattern::Matcher>& m, ov::matcher_pass_callback callback) {
            return std::make_shared<ov::pass::MatcherPass>(m, callback);
        }),
        py::arg("matcher"),
        py::arg("callback"),
        R"(
        Create MatcherPass from existing Matcher and callback objects.

        :param matcher: openvino.runtime.passes.Matcher with registered pattern.
        :type matcher: openvino.runtime.passes.Matcher

        :param callback: Function that performs transformation on the matched nodes.
        :type callback: function

        :return: created openvino.runtime.passes.MatcherPass instance.
        :rtype: openvino.runtime.passes.MatcherPass
    )");

    matcher_pass.def("apply",
                     &ov::pass::MatcherPass::apply,
                     py::arg("node"),
                     R"(
                     Execute MatcherPass on given Node.

                     :return: callback return code.
                     :rtype: bool
    )");

    matcher_pass.def("register_new_node",
                     &ov::pass::MatcherPass::register_new_node_,
                     py::arg("node"),
                     R"(
                     Register node for additional pattern matching.

                     :param node: openvino.runtime.Node for matching.
                     :type node: openvino.runtime.Node

                     :return: registered node instance
                     :rtype: openvino.runtime.Node
    )");

    matcher_pass.def("register_matcher",
                     static_cast<void (ov::pass::MatcherPass::*)(const std::shared_ptr<ov::pass::pattern::Matcher>&,
                                                                 const ov::graph_rewrite_callback& callback)>(
                         &PyMatcherPass::py_register_matcher),
                     py::arg("matcher"),
                     py::arg("callback"),
                     R"(
                     Initialize matcher and callback for further execution.

                     :param matcher: openvino.runtime.passes.Matcher with registered pattern.
                     :type matcher: openvino.runtime.passes.Matcher

                     :param callback: Function that performs transformation on the matched nodes.
                     :type callback: function
    )");
}
