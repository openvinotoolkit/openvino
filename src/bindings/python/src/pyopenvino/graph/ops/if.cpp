// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/if.hpp"

#include <string>

#include "ngraph/log.hpp"
#include "openvino/core/node.hpp"
#include "openvino/op/util/multi_subgraph_base.hpp"
#include "pyopenvino/graph/ops/util/multisubgraph.hpp"

namespace py = pybind11;

void regclass_graph_op_If(py::module m) {
    py::class_<ov::op::v8::If, std::shared_ptr<ov::op::v8::If>, ov::Node> cls(m, "if_op");
    cls.doc() = "openvino.impl.op.If wraps ov::op::v0::If";
    cls.def(py::init<>());
    cls.def(py::init<const ov::Output<ov::Node>&>(), py::arg("execution_condition"));
    cls.def(py::init([](const std::shared_ptr<ov::Node>& execution_condition) {
                if (MultiSubgraphHelpers::is_constant_or_parameter(execution_condition)) {
                    return std::make_shared<ov::op::v8::If>(execution_condition->output(0));
                } else {
                    NGRAPH_WARN << "Please specify execution_condition as Constant or Parameter. Default If() "
                                   "constructor was applied.";
                    return std::make_shared<ov::op::v8::If>();
                }
            }),
            py::arg("execution_condition"));
    cls.def("get_else_body", &ov::op::v8::If::get_else_body);
    cls.def("set_then_body", &ov::op::v8::If::set_then_body, py::arg("body"));
    cls.def("set_else_body", &ov::op::v8::If::set_else_body, py::arg("body"));
    cls.def("set_input",
            &ov::op::v8::If::set_input,
            py::arg("value"),
            py::arg("then_parameter"),
            py::arg("else_parameter"));
    cls.def("set_output", &ov::op::v8::If::set_output, py::arg("then_result"), py::arg("else_result"));
    cls.def("get_function", &ov::op::util::MultiSubGraphOp::get_function, py::arg("index"));
    cls.def("set_function", &ov::op::util::MultiSubGraphOp::set_function, py::arg("index"), py::arg("func"));

    cls.def(
        "set_input_descriptions",
        [](const std::shared_ptr<ov::op::v8::If>& self, int index, const py::list& inputs) {
            self->set_input_descriptions(index, MultiSubgraphHelpers::list_to_input_descriptor(inputs));
        },
        py::arg("index"),
        py::arg("inputs"));

    cls.def(
        "set_output_descriptions",
        [](const std::shared_ptr<ov::op::v8::If>& self, int index, const py::list& outputs) {
            self->set_output_descriptions(index, MultiSubgraphHelpers::list_to_output_descriptor(outputs));
        },
        py::arg("index"),
        py::arg("outputs"));

    cls.def(
        "get_output_descriptions",
        [](const std::shared_ptr<ov::op::v8::If>& self, int index) {
            py::list result;

            for (const auto& out_desc : self->get_output_descriptions(index)) {
                result.append(out_desc);
            }

            return result;
        },
        py::arg("index"));

    cls.def(
        "get_input_descriptions",
        [](const std::shared_ptr<ov::op::v8::If>& self, int index) {
            py::list result;

            for (const auto& in_desc : self->get_input_descriptions(index)) {
                result.append(in_desc);
            }

            return result;
        },
        py::arg("index"));
}
