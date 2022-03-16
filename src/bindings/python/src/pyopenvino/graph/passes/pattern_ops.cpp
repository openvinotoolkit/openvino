// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"

#include "ngraph/opsets/opset.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include <iterator>
#include <sstream>
#include <string>

#include "pyopenvino/graph/passes/pattern_ops.hpp"


ov::NodeTypeInfo get_type(const std::string & type_name) {
    // Supported types: opsetX.OpName or opsetX::OpName
    std::string opset_type;
    auto it = type_name.cbegin();
    while(it != type_name.cend() && *it != '.' && *it != ':') {
        opset_type += *it;
        ++it;
    }

    // Skip delimiter
    while (it != type_name.cend() && *it == '.' || *it == ':') {
        ++it;
    }

    // Get operation type name
    std::string operation_type(it, type_name.end());

    // TODO: create generic opset factory in Core so it can be reused
    const std::unordered_map<std::string, std::function<const ngraph::OpSet&()>> get_opset {
            {"opset1", ngraph::get_opset1},
            {"opset2", ngraph::get_opset2},
            {"opset3", ngraph::get_opset3},
            {"opset4", ngraph::get_opset4},
            {"opset5", ngraph::get_opset5},
            {"opset6", ngraph::get_opset6},
            {"opset7", ngraph::get_opset7},
            {"opset8", ngraph::get_opset8},
    };

    if (!get_opset.count(opset_type)) {
        throw std::runtime_error("Unsupported opset type: " + opset_type);
    }

    const ngraph::OpSet& m_opset = get_opset.at(opset_type)();
    if (!m_opset.contains_type(operation_type)) {
        throw std::runtime_error("Unrecognized operation type: " + operation_type);
    }

    return m_opset.create(operation_type)->get_type_info();
}

std::vector<ov::NodeTypeInfo> get_types(const std::vector<std::string> & type_names) {
    std::vector<ov::NodeTypeInfo> types;
    for (const auto & type_name : type_names) {
        types.emplace_back(get_type(type_name));
    }
    return types;
}

using Predicate = const ov::pass::pattern::op::ValuePredicate;

void reg_pattern_wrap_type(py::module m) {
    py::class_<ov::pass::pattern::op::WrapType,
               std::shared_ptr<ov::pass::pattern::op::WrapType>,
               ov::Node> wrap_type(m, "WrapType");
    wrap_type.doc() = "openvino.runtime.passes.WrapType wraps ov::pass::pattern::op::WrapType";

    wrap_type.def(py::init([](const std::string& name) {
        return std::make_shared<ov::pass::pattern::op::WrapType>(get_type(name));
    }));

    wrap_type.def(py::init([](const std::string& name, const Predicate& pred) {
        return std::make_shared<ov::pass::pattern::op::WrapType>(get_type(name), pred);
    }));

    wrap_type.def(py::init([](const std::string& name, const ov::Output<ov::Node>& input) {
        return std::make_shared<ov::pass::pattern::op::WrapType>(get_type(name), nullptr, ov::OutputVector{input});
    }));

    wrap_type.def(py::init([](const std::string& name, const std::shared_ptr<ov::Node>& input) {
        return std::make_shared<ov::pass::pattern::op::WrapType>(get_type(name), nullptr, ov::OutputVector{input});
    }));

    wrap_type.def(py::init([](const std::string& name, const ov::Output<ov::Node>& input, const Predicate& pred) {
        return std::make_shared<ov::pass::pattern::op::WrapType>(get_type(name), pred, ov::OutputVector{input});
    }));

    wrap_type.def(py::init([](const std::string& name, const std::shared_ptr<ov::Node>& input, const Predicate& pred) {
        return std::make_shared<ov::pass::pattern::op::WrapType>(get_type(name), pred, ov::OutputVector{input});
    }));

    wrap_type.def(py::init([](const std::string& name, const ov::OutputVector& inputs) {
        return std::make_shared<ov::pass::pattern::op::WrapType>(get_type(name), nullptr, inputs);
    }));

    wrap_type.def(py::init([](const std::string& name, const ov::NodeVector& inputs) {
        return std::make_shared<ov::pass::pattern::op::WrapType>(get_type(name), nullptr, ov::as_output_vector(inputs));
    }));

    wrap_type.def(py::init([](const std::string& name, const ov::OutputVector& inputs, const Predicate& pred) {
        return std::make_shared<ov::pass::pattern::op::WrapType>(get_type(name), pred, inputs);
    }));

    wrap_type.def(py::init([](const std::string& name, const ov::NodeVector& inputs, const Predicate& pred) {
        return std::make_shared<ov::pass::pattern::op::WrapType>(get_type(name), pred, ov::as_output_vector(inputs));
    }));

    wrap_type.def(py::init([](const std::vector<std::string>& names) {
        return std::make_shared<ov::pass::pattern::op::WrapType>(get_types(names));
    }));

    wrap_type.def(py::init([](const std::vector<std::string>& names, const Predicate& pred) {
        return std::make_shared<ov::pass::pattern::op::WrapType>(get_types(names), pred);
    }));

    wrap_type.def(py::init([](const std::vector<std::string>& names, const ov::Output<ov::Node>& input) {
        return std::make_shared<ov::pass::pattern::op::WrapType>(get_types(names), nullptr, ov::OutputVector{input});
    }));

    wrap_type.def(py::init([](const std::vector<std::string>& names, const std::shared_ptr<ov::Node>& input) {
        return std::make_shared<ov::pass::pattern::op::WrapType>(get_types(names), nullptr, ov::OutputVector{input});
    }));

    wrap_type.def(py::init([](const std::vector<std::string>& names, const ov::Output<ov::Node>& input, const Predicate& pred) {
        return std::make_shared<ov::pass::pattern::op::WrapType>(get_types(names), pred, ov::OutputVector{input});
    }));

    wrap_type.def(py::init([](const std::vector<std::string>& names, const std::shared_ptr<ov::Node>& input, const Predicate& pred) {
        return std::make_shared<ov::pass::pattern::op::WrapType>(get_types(names), pred, ov::OutputVector{input});
    }));

    wrap_type.def(py::init([](const std::vector<std::string>& names, const ov::OutputVector& input_values) {
        return std::make_shared<ov::pass::pattern::op::WrapType>(get_types(names), nullptr, input_values);
    }));

    wrap_type.def(py::init([](const std::vector<std::string>& names, const ov::NodeVector& input_values) {
        return std::make_shared<ov::pass::pattern::op::WrapType>(get_types(names), nullptr, ov::as_output_vector(input_values));
    }));

    wrap_type.def(py::init([](const std::vector<std::string>& names, const ov::OutputVector& input_values, const Predicate& pred) {
        return std::make_shared<ov::pass::pattern::op::WrapType>(get_types(names), pred, input_values);
    }));

    wrap_type.def(py::init([](const std::vector<std::string>& names, const ov::NodeVector& input_values, const Predicate& pred) {
        return std::make_shared<ov::pass::pattern::op::WrapType>(get_types(names), pred, ov::as_output_vector(input_values));
    }));
}

void reg_pattern_or(py::module m) {
    py::class_<ov::pass::pattern::op::Or,
               std::shared_ptr<ov::pass::pattern::op::Or>,
               ov::Node> or_type(m, "Or");
    or_type.doc() = "openvino.runtime.passes.Or wraps ov::pass::pattern::op::Or";

    or_type.def(py::init([](const ov::OutputVector& inputs) {
        return std::make_shared<ov::pass::pattern::op::Or>(inputs);
    }));

    or_type.def(py::init([](const ov::NodeVector& inputs) {
        return std::make_shared<ov::pass::pattern::op::Or>(ov::as_output_vector(inputs));
    }));
}

void reg_pattern_any_input(py::module m) {
    py::class_<ov::pass::pattern::op::Label,
               std::shared_ptr<ov::pass::pattern::op::Label>,
               ov::Node> any_input(m, "AnyInput");
    any_input.doc() = "openvino.runtime.passes.AnyInput wraps ov::pass::pattern::op::Label";

    any_input.def(py::init([]() {
        return std::make_shared<ov::pass::pattern::op::Label>();
    }));

    any_input.def(py::init([](const Predicate& pred) {
        return std::make_shared<ov::pass::pattern::op::Label>(ov::element::dynamic, ov::PartialShape::dynamic(), pred);
    }));
}

void reg_predicates(py::module m) {
    m.def("consumers_count", &ov::pass::pattern::consumers_count);
    m.def("has_static_dim", &ov::pass::pattern::has_static_dim);
    m.def("has_static_dims", &ov::pass::pattern::has_static_dims);
    m.def("has_static_shape", &ov::pass::pattern::has_static_shape);
    m.def("has_static_rank", &ov::pass::pattern::has_static_rank);
    m.def("rank_equals", &ov::pass::pattern::rank_equals);
    m.def("type_matches", &ov::pass::pattern::type_matches);
    m.def("type_matches_any", &ov::pass::pattern::type_matches_any);
}

void reg_pass_pattern_ops(py::module m) {
    reg_pattern_any_input(m);
    reg_pattern_wrap_type(m);
    reg_pattern_or(m);
    reg_predicates(m);
}

