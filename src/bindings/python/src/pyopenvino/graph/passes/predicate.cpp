// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/graph/passes/predicate.hpp"

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "openvino/pass/pattern/op/predicate.hpp"
#include "pyopenvino/core/common.hpp"

using ValuePredicate = const ov::pass::pattern::op::ValuePredicate;
using SymbolPredicate = const std::function<bool(ov::pass::pattern::PatternSymbolMap&, const ov::Output<ov::Node>&)>;

static void reg_pattern_op_predicate(py::module m) {
    py::class_<ov::pass::pattern::op::Predicate, std::shared_ptr<ov::pass::pattern::op::Predicate>> predicate(
        m,
        "Predicate");
    predicate.doc() = "openvino.passes.Predicate wraps ov::pass::pattern::op::Predicate";

    predicate.def(py::init([]() {
                      return std::make_shared<ov::pass::pattern::op::Predicate>();
                  }),
                  R"(
                  Create default Predicate which always returns true.
    )");

    predicate.def(py::init([](ValuePredicate pred) {
                      return std::make_shared<ov::pass::pattern::op::Predicate>(pred);
                  }),
                  py::arg("predicate"),
                  R"(
                  Create Predicate from a given function.

                  :param predicate: function (Output<Node> -> bool)
                  :type predicate: Callable
    )");

    predicate.def(py::init([](SymbolPredicate pred) {
                      return std::make_shared<ov::pass::pattern::op::Predicate>(pred);
                  }),
                  py::arg("predicate"),
                  R"(
                  Create Predicate from a given function.

                  :param predicate: function (PatternSymbolMap&, Output<Node> -> bool)
                  :type predicate: Callable
    )");
}

static void reg_pattern_symbol_value(py::module m) {
    py::class_<ov::pass::pattern::PatternSymbolValue, std::shared_ptr<ov::pass::pattern::PatternSymbolValue>> value(
        m,
        "PatternSymbolValue");
    value.doc() = "openvino.passes.PatternSymbolValue wraps ov::pass::pattern::PatternSymbolValue";

    value.def(py::init([](const std::shared_ptr<ov::Symbol>& s) {
                  return ov::pass::pattern::PatternSymbolValue(s);
              }),
              py::arg("value"),
              R"(
        Create PatternSymbolValue with the given value.

        :param value: symbol to keep as pattern value
        :type value: openvino.Symbol
    )");

    value.def(py::init([](const int64_t& s) {
                  return ov::pass::pattern::PatternSymbolValue(s);
              }),
              py::arg("value"),
              R"(
        Create PatternSymbolValue with the given value.

        :param value: integer to keep as a pattern value
        :type value: int
    )");

    value.def(py::init([](const double& s) {
                  return ov::pass::pattern::PatternSymbolValue(s);
              }),
              py::arg("value"),
              R"(
        Create PatternSymbolValue with the given value.

        :param value: float to keep as a pattern value
        :type value: float
    )");

    value.def(py::init([](const std::vector<ov::pass::pattern::PatternSymbolValue>& s) {
                  return ov::pass::pattern::PatternSymbolValue(s);
              }),
              py::arg("value"),
              R"(
        Create PatternSymbolValue with the given value.

        :param value: list of values representing a group of pattern values
        :type value: List[PatternSymbolValue]
    )");

    value.def("is_dynamic", &ov::pass::pattern::PatternSymbolValue::is_dynamic);
    value.def("is_static", &ov::pass::pattern::PatternSymbolValue::is_static);
    value.def("is_group", &ov::pass::pattern::PatternSymbolValue::is_group);
    value.def("is_integer", &ov::pass::pattern::PatternSymbolValue::is_integer);
    value.def("is_double", &ov::pass::pattern::PatternSymbolValue::is_double);

    value.def("i", &ov::pass::pattern::PatternSymbolValue::i);
    value.def("d", &ov::pass::pattern::PatternSymbolValue::d);
    value.def("s", &ov::pass::pattern::PatternSymbolValue::s);
    value.def("g", &ov::pass::pattern::PatternSymbolValue::g);

    value.def(
        "__eq__",
        [](const ov::pass::pattern::PatternSymbolValue& lhs, const ov::pass::pattern::PatternSymbolValue& rhs) {
            return lhs == rhs;
        },
        py::is_operator());
}

void reg_passes_predicate(py::module m) {
    reg_pattern_op_predicate(m);
    reg_pattern_symbol_value(m);
}
