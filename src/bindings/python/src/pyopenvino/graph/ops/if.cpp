// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/if.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <string>

#include "ngraph/log.hpp"
#include "openvino/core/node.hpp"
#include "openvino/op/util/multi_subgraph_base.hpp"
#include "openvino/op/util/sub_graph_base.hpp"
#include "pyopenvino/graph/ops/if.hpp"
#include "pyopenvino/graph/ops/util/multisubgraph.hpp"

namespace py = pybind11;

using MultiSubgraphInputDescriptionVector = ov::op::util::MultiSubGraphOp::MultiSubgraphInputDescriptionVector;
using MultiSubgraphOutputDescriptionVector = ov::op::util::MultiSubGraphOp::MultiSubgraphOutputDescriptionVector;

// PYBIND11_MAKE_OPAQUE(MultiSubgraphInputDescriptionVector);
PYBIND11_MAKE_OPAQUE(MultiSubgraphInputDescriptionVector);
PYBIND11_MAKE_OPAQUE(std::vector<ov::op::util::MultiSubGraphOp::OutputDescription::Ptr>);
PYBIND11_MAKE_OPAQUE(std::vector<int>);

MultiSubgraphInputDescriptionVector list_to_input_descriptor(const py::list& inputs) {
    std::vector<ov::op::util::MultiSubGraphOp::InputDescription::Ptr> result;

    for (auto& in_desc : inputs) {
        if (py::isinstance<ov::op::util::MultiSubGraphOp::SliceInputDescription>(in_desc)) {
            auto casted = in_desc.cast<std::shared_ptr<ov::op::util::MultiSubGraphOp::SliceInputDescription>>();
            result.emplace_back(casted);
        } else if (py::isinstance<ov::op::util::MultiSubGraphOp::MergedInputDescription>(in_desc)) {
            auto casted = in_desc.cast<std::shared_ptr<ov::op::util::MultiSubGraphOp::MergedInputDescription>>();
            result.emplace_back(casted);
        } else if (py::isinstance<ov::op::util::MultiSubGraphOp::InvariantInputDescription>(in_desc)) {
            auto casted = in_desc.cast<std::shared_ptr<ov::op::util::MultiSubGraphOp::InvariantInputDescription>>();
            result.emplace_back(casted);
        } else {
            throw py::type_error("Incompatible InputDescription type, following are supported: SliceInputDescription, "
                                 "MergedInputDescription and InvariantInputDescription.");
        }
    }

    std::cout << "incoming inputs size: " << result.size() << std::endl;

    return result;
}

const py::list input_descriptor_to_list(MultiSubgraphInputDescriptionVector& inputs) {
    py::list result;
    std::cout << "inputs.size(): " << inputs.size() << std::endl;

    for (auto& in_desc : inputs) {
        result.append(in_desc.get());
    }

    return result;
}

MultiSubgraphOutputDescriptionVector list_to_output_descriptor(py::list& outputs) {
    std::vector<ov::op::util::MultiSubGraphOp::OutputDescription::Ptr> result(py::len(outputs));
    result.clear();

    for (auto& out_desc : outputs) {
        if (py::isinstance<ov::op::util::MultiSubGraphOp::ConcatOutputDescription>(out_desc)) {
            auto casted = out_desc.cast<std::shared_ptr<ov::op::util::MultiSubGraphOp::ConcatOutputDescription>>();
            result.emplace_back(casted);
        } else if (py::isinstance<ov::op::util::MultiSubGraphOp::BodyOutputDescription>(out_desc)) {
            auto casted = out_desc.cast<std::shared_ptr<ov::op::util::MultiSubGraphOp::BodyOutputDescription>>();
            result.emplace_back(casted);
        } else {
            throw py::type_error("Incompatible OutputDescription type, following are supported: "
                                 "ConcatOutputDescription and BodyOutputDescription.");
        }
    }

    return result;
}

void regclass_MultiSubgraphInputDescriptionVector(py::module m) {
    py::bind_vector<MultiSubgraphInputDescriptionVector>(m, "MultiSubgraphInputDescriptionVector");
}

void regclass_graph_op_If(py::module m) {
    py::class_<ov::op::v8::If, std::shared_ptr<ov::op::v8::If>, ov::Node> cls(m, "if_op");
    cls.doc() = "openvino.impl.op.If wraps ov::op::v0::If";
    cls.def(py::init<>());
    cls.def(py::init<const ov::Output<ov::Node>&>(), py::arg("execution_condition"));
    cls.def(py::init([](ov::Node& execution_condition) {
                auto name = std::string(execution_condition.get_type_name());
                if (name == "Constant" || name == "Parameter") {
                    return ov::op::v8::If(execution_condition.output(0));
                } else {
                    NGRAPH_WARN << "Please specify execution_condition as Constant or Parameter. Default If() "
                                   "constructor was applied.";
                    return ov::op::v8::If();
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
        [](ov::op::v8::If& self, int& index, py::list& inputs) {
            self.set_input_descriptions(index, list_to_input_descriptor(inputs));
        },
        py::arg("index"),
        py::arg("inputs"));

    cls.def(
        "set_output_descriptions",
        [](ov::op::v8::If& self, int& index, py::list outputs) {
            self.set_output_descriptions(index, list_to_output_descriptor(outputs));
        },
        py::arg("index"),
        py::arg("outputs"));

    cls.def(
        "get_output_descriptions",
        [](ov::op::v8::If& self, int& index) {
            py::list result;
            auto outputs = self.get_output_descriptions(index);

            for (const auto& out_desc : outputs) {
                result.append(out_desc.get());
            }

            return result;
        },
        py::arg("index"));

    cls.def(
        "get_input_descriptions",
        [](ov::op::v8::If& self, int& index) -> py::list {
            py::list result;

            for (const auto& in_desc : self.get_input_descriptions(index)) {
                result.append(in_desc.get());
            }

            return result;
        },
        py::arg("index"));
}
