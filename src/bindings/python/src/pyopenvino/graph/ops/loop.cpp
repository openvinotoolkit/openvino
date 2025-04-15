// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/graph/ops/loop.hpp"

#include <string>

#include "openvino/core/node.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/util/log.hpp"
#include "pyopenvino/core/common.hpp"
#include "pyopenvino/graph/ops/util/multisubgraph.hpp"
#include "pyopenvino/utils/utils.hpp"

namespace py = pybind11;

void regclass_graph_op_Loop(py::module m) {
    py::class_<ov::op::v5::Loop, std::shared_ptr<ov::op::v5::Loop>, ov::Node> cls(m, "loop");
    cls.doc() = "openvino.impl.op.Loop wraps ov::op::v0::Loop";
    cls.def(py::init<>());
    cls.def(py::init<const ov::Output<ov::Node>&, const ov::Output<ov::Node>&>(),
            py::arg("trip_count"),
            py::arg("execution_condition"));

    cls.def(
        py::init([](const std::shared_ptr<ov::Node>& trip_count, const std::shared_ptr<ov::Node>& execution_condition) {
            return std::make_shared<ov::op::v5::Loop>(trip_count, execution_condition);
        }),
        py::arg("trip_count"),
        py::arg("execution_condition"));

    cls.def("set_invariant_input", &ov::op::v5::Loop::set_invariant_input, py::arg("body_parameter"), py::arg("value"));
    cls.def("get_iter_value", &ov::op::v5::Loop::get_iter_value, py::arg("body_value"), py::arg("iteration") = -1);
    cls.def("get_num_iterations", &ov::op::v5::Loop::get_num_iterations);

    cls.def("get_concatenated_slices",
            &ov::op::v5::Loop::get_concatenated_slices,
            py::arg("value"),
            py::arg("start"),
            py::arg("stride"),
            py::arg("part_size"),
            py::arg("end"),
            py::arg("axis"));

    cls.def(
        "set_special_body_ports",
        [](const std::shared_ptr<ov::op::v5::Loop>& self, const py::list& special_body_ports) {
            if (py::len(special_body_ports) == 2) {
                ov::op::v5::Loop::SpecialBodyPorts bodyPorts;
                bodyPorts.current_iteration_input_idx = special_body_ports[0].cast<int64_t>();
                bodyPorts.body_condition_output_idx = special_body_ports[1].cast<int64_t>();
                self->set_special_body_ports(bodyPorts);
            } else {
                throw py::type_error("Invalid SpecialBodyPorts size, it shall contain two parameters: "
                                     "body_condition_output_idx and current_iteration_input_idx.");
            }
        },
        py::arg("special_body_ports"));

    cls.def("get_special_body_ports", [](const std::shared_ptr<ov::op::v5::Loop>& self) {
        py::list result;
        auto bodyPorts = self->get_special_body_ports();

        result.append(bodyPorts.current_iteration_input_idx);
        result.append(bodyPorts.body_condition_output_idx);

        return result;
    });

    cls.def("set_sliced_input",
            &ov::op::v5::Loop::set_sliced_input,
            py::arg("parameter"),
            py::arg("value"),
            py::arg("start"),
            py::arg("stride"),
            py::arg("part_size"),
            py::arg("end"),
            py::arg("axis"));

    cls.def("set_merged_input",
            &ov::op::v5::Loop::set_merged_input,
            py::arg("body_parameter"),
            py::arg("initial_value"),
            py::arg("successive_value"));

    cls.def("get_function", [](const std::shared_ptr<ov::op::v5::Loop>& self) {
        auto model = self->get_function();
        py::type model_class = py::module_::import("openvino").attr("Model");
        return model_class(py::cast(model));
    });

    cls.def(
        "set_function",
        [](const std::shared_ptr<ov::op::v5::Loop>& self, const py::object& ie_api_model) {
            const auto func = Common::utils::convert_to_model(ie_api_model);
            self->set_function(func);
        },
        py::arg("func"));

    cls.def("get_output_descriptions", [](const std::shared_ptr<ov::op::v5::Loop>& self) {
        py::list result;

        for (const auto& out_desc : self->get_output_descriptions()) {
            result.append(out_desc);
        }

        return result;
    });

    cls.def("get_input_descriptions", [](const std::shared_ptr<ov::op::v5::Loop>& self) {
        py::list result;

        for (const auto& in_desc : self->get_input_descriptions()) {
            result.append(in_desc);
        }

        return result;
    });

    cls.def(
        "set_input_descriptions",
        [](const std::shared_ptr<ov::op::v5::Loop>& self, const py::list& inputs) {
            self->set_input_descriptions(0, MultiSubgraphHelpers::list_to_input_descriptor(inputs));
        },
        py::arg("inputs"));

    cls.def(
        "set_output_descriptions",
        [](const std::shared_ptr<ov::op::v5::Loop>& self, const py::list& outputs) {
            self->set_output_descriptions(0, MultiSubgraphHelpers::list_to_output_descriptor(outputs));
        },
        py::arg("outputs"));

    cls.def("__repr__", [](const ov::op::v5::Loop& self) {
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
