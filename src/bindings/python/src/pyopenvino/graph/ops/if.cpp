// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/if.hpp"

#include <string>

#include "openvino/core/node.hpp"
#include "openvino/op/util/multi_subgraph_base.hpp"
#include "openvino/util/log.hpp"
#include "pyopenvino/core/common.hpp"
#include "pyopenvino/graph/ops/if.hpp"
#include "pyopenvino/graph/ops/util/multisubgraph.hpp"
#include "pyopenvino/utils/utils.hpp"

namespace py = pybind11;

void regclass_graph_op_If(py::module m) {
    py::class_<ov::op::v8::If, std::shared_ptr<ov::op::v8::If>, ov::Node> cls(m, "if_op");
    cls.doc() = "openvino.impl.op.If wraps ov::op::v0::If";
    cls.def(py::init<>());
    cls.def(py::init<const ov::Output<ov::Node>&>(),
            py::arg("execution_condition"),
            R"(
            Constructs If with condition.

            :param execution_condition: condition node.
            :type execution_condition: openvino.Output

            :rtype: openvino.impl.op.If
        )");

    cls.def(py::init([](const std::shared_ptr<ov::Node>& execution_condition) {
                if (MultiSubgraphHelpers::is_constant_or_parameter(execution_condition)) {
                    return std::make_shared<ov::op::v8::If>(execution_condition->output(0));
                } else {
                    OPENVINO_WARN("Please specify execution_condition as Constant or Parameter. Default If() "
                                  "constructor was applied.");
                    return std::make_shared<ov::op::v8::If>();
                }
            }),
            py::arg("execution_condition"),
            R"(
            Constructs If with condition.

            :param execution_condition: condition node.
            :type execution_condition: openvino.Node

            :rtype: openvino.impl.op.If
        )");

    cls.def(
        "get_then_body",
        [](ov::op::v8::If& self) {
            auto model = self.get_then_body();
            py::type model_class = py::module_::import("openvino").attr("Model");
            return model_class(py::cast(model));
        },
        R"(
            Gets then_body as Model object.

            :return: then_body as Model object.
            :rtype: openvino.Model
        )");

    cls.def(
        "get_else_body",
        [](ov::op::v8::If& self) {
            auto model = self.get_else_body();
            py::type model_class = py::module_::import("openvino").attr("Model");
            return model_class(py::cast(model));
        },
        R"(
            Gets else_body as Model object.

            :return: else_body as Model object.
            :rtype: openvino.Model
        )");

    cls.def(
        "set_then_body",
        [](const std::shared_ptr<ov::op::v8::If>& self, const py::object& ie_api_model) {
            const auto body = Common::utils::convert_to_model(ie_api_model);
            return self->set_then_body(body);
        },
        py::arg("body"),
        R"(
            Sets new Model object as new then_body.

            :param body: new body for 'then' branch.
            :type body: openvino.Model

            :rtype: None
        )");

    cls.def(
        "set_else_body",
        [](const std::shared_ptr<ov::op::v8::If>& self, const py::object& ie_api_model) {
            const auto body = Common::utils::convert_to_model(ie_api_model);
            return self->set_else_body(body);
        },
        py::arg("body"),
        R"(
            Sets new Model object as new else_body.

            :param body: new body for 'else' branch.
            :type body: openvino.Model

            :rtype: None
        )");

    cls.def("set_input",
            &ov::op::v8::If::set_input,
            py::arg("value"),
            py::arg("then_parameter"),
            py::arg("else_parameter"),
            R"(
            Sets new input to the operation associated with parameters of each sub-graphs.

            :param value: input to operation.
            :type value: openvino.Output

            :param then_result: parameter for then_body or nullptr.
            :type then_result: openvino.Node

            :param else_result: parameter for else_body or nullptr.
            :type else_result: openvino.Node

            :rtype: None
        )");

    cls.def("set_output",
            &ov::op::v8::If::set_output,
            py::arg("then_result"),
            py::arg("else_result"),
            R"(
            Sets new output from the operation associated with results of each sub-graphs.

            :param then_result: result from then_body.
            :type then_result: op.Result

            :param else_result: result from else_body.
            :type else_result: op.Result

            :return: output from operation.
            :rtype: openvino.Output
        )");

    cls.def(
        "get_function",
        [](ov::op::v8::If& self, size_t index) {
            auto model = self.get_function(index);
            py::type model_class = py::module_::import("openvino").attr("Model");
            return model_class(py::cast(model));
        },
        py::arg("index"),
        R"(
            Gets internal sub-graph by index in MultiSubGraphOp.

            :param index: sub-graph's index in op.
            :type index: int
            
            :return: Model with sub-graph.
            :rtype: openvino.Model
        )");

    cls.def(
        "set_function",
        [](const std::shared_ptr<ov::op::v8::If>& self, int index, const py::object& ie_api_model) {
            const auto func = Common::utils::convert_to_model(ie_api_model);
            self->set_function(index, func);
        },
        py::arg("index"),
        py::arg("func"),
        R"(
            Adds sub-graph to MultiSubGraphOp.

            :param index: index of new sub-graph.
            :type index: int

            :param func: func new sub_graph as a Model.
            :type func: openvino.Model

            :rtype: None
        )");

    cls.def(
        "set_input_descriptions",
        [](const std::shared_ptr<ov::op::v8::If>& self, int index, const py::list& inputs) {
            self->set_input_descriptions(index, MultiSubgraphHelpers::list_to_input_descriptor(inputs));
        },
        py::arg("index"),
        py::arg("inputs"),
        R"(
            Sets list with connections between operation inputs and internal sub-graph parameters.

            :param index: index of internal sub-graph.
            :type index: int

            :param inputs: list of input descriptions.
            :type inputs: list[Union[openvino.op.util.MergedInputDescription,
                                     openvino.op.util.InvariantInputDescription,
                                     openvino.op.util.SliceInputDescription]]

            :rtype: None
        )");

    cls.def(
        "set_output_descriptions",
        [](const std::shared_ptr<ov::op::v8::If>& self, int index, const py::list& outputs) {
            self->set_output_descriptions(index, MultiSubgraphHelpers::list_to_output_descriptor(outputs));
        },
        py::arg("index"),
        py::arg("outputs"),
        R"(
            Sets list with connections between operation outputs and internal sub-graph parameters.

            :param index: index of internal sub-graph.
            :type index: int

            :param outputs: list of output descriptions.
            :type outputs: list[Union[openvino.op.util.BodyOutputDescription,
                                      openvino.op.util.ConcatOutputDescription]]

            :rtype: None
        )");

    cls.def(
        "get_output_descriptions",
        [](const std::shared_ptr<ov::op::v8::If>& self, int index) {
            py::list result;

            for (const auto& out_desc : self->get_output_descriptions(index)) {
                result.append(out_desc);
            }

            return result;
        },
        py::arg("index"),
        R"(
            Gets list with connections between operation outputs and internal sub-graph parameters.

            :param index: index of internal sub-graph.
            :type index: int

            :return: list of output descriptions.
            :rtype: list[Union[openvino.op.util.BodyOutputDescription,
                              openvino.op.util.ConcatOutputDescription]]
        )");

    cls.def(
        "get_input_descriptions",
        [](const std::shared_ptr<ov::op::v8::If>& self, int index) {
            py::list result;

            for (const auto& in_desc : self->get_input_descriptions(index)) {
                result.append(in_desc);
            }

            return result;
        },
        py::arg("index"),
        R"(
            Gets list with connections between operation inputs and internal sub-graph parameters.

            :param index: index of internal sub-graph.
            :type index: int

            :return: list of input descriptions.
            :rtype: list[Union[openvino.op.util.MergedInputDescription,
                               openvino.op.util.InvariantInputDescription,
                               openvino.op.util.SliceInputDescription]]
        )");

    cls.def("__repr__", [](const ov::op::v8::If& self) {
        std::stringstream shapes_ss;
        for (size_t i = 0; i < self.get_output_size(); ++i) {
            if (i > 0) {
                shapes_ss << ", ";
            }
            shapes_ss << self.get_output_partial_shape(i);
        }
        return "<" + Common::get_class_name(self) + ": '" + self.get_friendly_name() + "' (" + shapes_ss.str() + ")>";
    });
}
