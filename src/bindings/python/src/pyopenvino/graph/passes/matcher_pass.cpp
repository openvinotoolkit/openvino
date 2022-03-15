// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/pattern/matcher.hpp"  // ov::pattern::Matcher
#include "openvino/pass/graph_rewrite.hpp"  // ov::pattern::Matcher
#include "openvino/pass/pattern/op/wrap_type.hpp"

#include "ngraph/opsets/opset.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include <iterator>
#include <sstream>
#include <string>
#include <transformations/serialize.hpp>

#include "pyopenvino/graph/passes/matcher_pass.hpp"

namespace py = pybind11;

void regclass_Matcher(py::module m) {
    py::class_<ov::pass::pattern::Matcher, std::shared_ptr<ov::pass::pattern::Matcher>> matcher(m, "Matcher");
    matcher.doc() = "openvino.impl.Matcher wraps ov::pass::pattern::Matcher";
    matcher.def(py::init([](const std::shared_ptr<ov::Node>& node,
                            const std::string& name) {
                    return std::make_shared<ov::pass::pattern::Matcher>(node, name);
                }),
                py::arg("node"),
                py::arg("name"),
                R"(
                    Create user-defined Function which is a representation of a model.
                    Parameters
                    ----------
                    results : List[op.Result]
                        List of results.
                    sinks : List[Node]
                        List of Nodes to be used as Sinks (e.g. Assign ops).
                    parameters : List[op.Parameter]
                        List of parameters.
                    name : str
                        String to set as function's friendly name.
                 )");

    matcher.def(py::init([](ov::Output<ov::Node> & output,
                            const std::string& name) {
                    return std::make_shared<ov::pass::pattern::Matcher>(output, name);
                }),
                py::arg("output"),
                py::arg("name"),
                R"(
                    Create user-defined Function which is a representation of a model.
                    Parameters
                    ----------
                    results : List[op.Result]
                        List of results.
                    sinks : List[Node]
                        List of Nodes to be used as Sinks (e.g. Assign ops).
                    parameters : List[op.Parameter]
                        List of parameters.
                    name : str
                        String to set as function's friendly name.
                 )");

    matcher.def("get_name", &ov::pass::pattern::Matcher::get_name);
    matcher.def("get_match_root", &ov::pass::pattern::Matcher::get_match_root);
    matcher.def("get_match_value", &ov::pass::pattern::Matcher::get_match_value);
    matcher.def("get_match_nodes", &ov::pass::pattern::Matcher::get_matched_nodes);
    matcher.def("get_match_values", static_cast<const ov::OutputVector& (ov::pass::pattern::Matcher::*)() const>(&ov::pass::pattern::Matcher::get_matched_values));
    matcher.def("get_pattern_value_map", &ov::pass::pattern::Matcher::get_pattern_value_map);
}

// Work around to expose protected methods
class PyMatcherPass: public ov::pass::MatcherPass {
public:
    using ov::pass::MatcherPass::MatcherPass;
    using ov::pass::MatcherPass::register_matcher;

    std::shared_ptr<ov::Node> register_new_node_(const std::shared_ptr<ov::Node>& node) {
        return register_new_node(node);
    }
};

void regclass_MatcherPass(py::module m) {
    py::class_<ov::pass::MatcherPass,
    std::shared_ptr<ov::pass::MatcherPass>,
    ov::pass::PassBase> matcher_pass(m, "MatcherPass");
    matcher_pass.doc() = "openvino.runtime.passses.MatcherPass wraps ov::pass::MatcherPass";
    matcher_pass.def(py::init<>());
    matcher_pass.def(py::init([](const std::shared_ptr<ov::pass::pattern::Matcher>& m,
                                 ov::matcher_pass_callback callback) {
                         return std::make_shared<ov::pass::MatcherPass>(m, callback);
                     }),
                     py::arg("m"),
                     py::arg("callback"),
                     R"(
                    Create user-defined Function which is a representation of a model.
                    Parameters
                    ----------
                    results : List[op.Result]
                        List of results.
                    sinks : List[Node]
                        List of Nodes to be used as Sinks (e.g. Assign ops).
                    parameters : List[op.Parameter]
                        List of parameters.
                    name : str
                        String to set as function's friendly name.
                 )");
    matcher_pass.def("apply", &ov::pass::MatcherPass::apply);
    matcher_pass.def("register_new_node", &ov::pass::MatcherPass::register_new_node_);
    matcher_pass.def("register_matcher", static_cast<void (ov::pass::MatcherPass::*)(const std::shared_ptr<ov::pass::pattern::Matcher>&,
                                                                                     const ov::graph_rewrite_callback& callback)>(&PyMatcherPass::register_matcher));
}

ov::NodeTypeInfo get_type(const std::string & type_name) {
    // TODO: allow to specify opset version
    const ngraph::OpSet& m_opset = ngraph::get_opset8();
    if (!m_opset.contains_type(type_name)) {
        throw std::runtime_error("Wrong pattern type:" +type_name + " in not in opset8");
    }
    return m_opset.create(type_name)->get_type_info();
}

std::vector<ov::NodeTypeInfo> get_types(const std::vector<std::string> & type_names) {
    std::vector<ov::NodeTypeInfo> types;
    for (const auto & type_name : type_names) {
        types.emplace_back(get_type(type_name));
    }
    return types;
}

void regclass_pass_patterns(py::module m) {
    py::class_<ov::pass::pattern::op::WrapType,
    std::shared_ptr<ov::pass::pattern::op::WrapType>,
    ov::Node> wrap_type(m, "WrapType");
    wrap_type.doc() = "openvino.impl.MatcherPass wraps ov::pass::MatcherPass";

    wrap_type.def(py::init([](std::string name) {
        return std::make_shared<ov::pass::pattern::op::WrapType>(get_type(name));
    }));

    wrap_type.def(py::init([](std::string name, const ov::Output<ov::Node>& input) {
        return std::make_shared<ov::pass::pattern::op::WrapType>(get_type(name), nullptr, ov::OutputVector{input});
    }));

    wrap_type.def(py::init([](std::string name, const ov::OutputVector& inputs) {
        return std::make_shared<ov::pass::pattern::op::WrapType>(get_type(name), nullptr, inputs);
    }));

    wrap_type.def(py::init([](std::vector<std::string> names) {
        return std::make_shared<ov::pass::pattern::op::WrapType>(get_types(names));
    }));

    wrap_type.def(py::init([](std::vector<std::string> names, const ov::Output<ov::Node>& input) {
        return std::make_shared<ov::pass::pattern::op::WrapType>(get_types(names), nullptr, ov::OutputVector{input});
    }));

    wrap_type.def(py::init([](std::vector<std::string> names, const ov::OutputVector& input_values) {
        return std::make_shared<ov::pass::pattern::op::WrapType>(get_types(names), nullptr, input_values);
    }));
}
