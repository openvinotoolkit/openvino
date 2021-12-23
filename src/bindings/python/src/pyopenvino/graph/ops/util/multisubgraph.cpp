// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/graph/ops/util/multisubgraph.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>

#include "openvino/core/node.hpp"
//#include "openvino/op/util/sub_graph_base.hpp"
#include "ngraph/log.hpp"

namespace py = pybind11;

class PyInputDescription : public ov::op::util::MultiSubGraphOp::InputDescription {
public:
    using ov::op::util::MultiSubGraphOp::InputDescription::InputDescription;
    std::shared_ptr<InputDescription> copy() const override {
        PYBIND11_OVERRIDE_PURE(std::shared_ptr<ov::op::util::MultiSubGraphOp::InputDescription>,
                               ov::op::util::MultiSubGraphOp::InputDescription,
                               copy, );
    }

    const type_info_t& get_type_info() const override {
        PYBIND11_OVERRIDE_PURE(type_info_t&, ov::op::util::MultiSubGraphOp::InputDescription, get_type_info, );
    }
};

class PyOutputDescription : public ov::op::util::MultiSubGraphOp::OutputDescription {
public:
    using ov::op::util::MultiSubGraphOp::OutputDescription::OutputDescription;
    std::shared_ptr<OutputDescription> copy() const override {
        PYBIND11_OVERRIDE_PURE(std::shared_ptr<ov::op::util::MultiSubGraphOp::OutputDescription>,
                               ov::op::util::MultiSubGraphOp::OutputDescription,
                               copy, );
    }

    const type_info_t& get_type_info() const override {
        PYBIND11_OVERRIDE_PURE(type_info_t&, ov::op::util::MultiSubGraphOp::OutputDescription, get_type_info, );
    }
};

void regclass_graph_op_util_MultiSubgraphOp(py::module m) {
    py::class_<ov::op::util::MultiSubGraphOp::InputDescription,
               std::shared_ptr<ov::op::util::MultiSubGraphOp::InputDescription>,
               PyInputDescription>
        input(m, "InputDescription");
    input.doc() =
        "ov::op::util::MultiSubGraphOp::InputDescription wraps ov::op::util::MultiSubGraphOp::InputDescription";
    input.def(py::init<>());
    input.def("copy", &ov::op::util::MultiSubGraphOp::InputDescription::copy);

    py::class_<ov::op::util::MultiSubGraphOp::SliceInputDescription,
               std::shared_ptr<ov::op::util::MultiSubGraphOp::SliceInputDescription>,
               ov::op::util::MultiSubGraphOp::InputDescription>(m, "SliceInputDescription")
        .def(py::init<>())
        .def(py::init<uint64_t&, uint64_t&, int64_t&, int64_t&, int64_t&, int64_t&, int64_t&>(),
             py::arg("input_index"),
             py::arg("body_parameter_index"),
             py::arg("start"),
             py::arg("stride"),
             py::arg("part_size"),
             py::arg("end"),
             py::arg("axis"))
        .def("copy", &ov::op::util::MultiSubGraphOp::SliceInputDescription::copy)
        .def("get_type_info", &ov::op::util::MultiSubGraphOp::SliceInputDescription::get_type_info)
        .def_readonly("input_index", &ov::op::util::MultiSubGraphOp::SliceInputDescription::m_input_index)
        .def_readonly("body_parameter_index",
                      &ov::op::util::MultiSubGraphOp::SliceInputDescription::m_body_parameter_index)
        .def_readonly("start", &ov::op::util::MultiSubGraphOp::SliceInputDescription::m_start)
        .def_readonly("stride", &ov::op::util::MultiSubGraphOp::SliceInputDescription::m_stride)
        .def_readonly("part_size", &ov::op::util::MultiSubGraphOp::SliceInputDescription::m_part_size)
        .def_readonly("end", &ov::op::util::MultiSubGraphOp::SliceInputDescription::m_end)
        .def_readonly("axis", &ov::op::util::MultiSubGraphOp::SliceInputDescription::m_axis);

    py::class_<ov::op::util::MultiSubGraphOp::MergedInputDescription,
               std::shared_ptr<ov::op::util::MultiSubGraphOp::MergedInputDescription>,
               ov::op::util::MultiSubGraphOp::InputDescription>
        merged(m, "MergedInputDescription");
    merged.doc() = "openvino.impl.op.util.MergedInputDescription wraps ov::op::util::MergedInputDescription";
    merged.def(py::init<>());
    merged.def(py::init<uint64_t&, uint64_t&, uint64_t&>(),
               py::arg("input_index"),
               py::arg("body_parameter_index"),
               py::arg("body_value_index"));
    merged.def("copy", &ov::op::util::MultiSubGraphOp::MergedInputDescription::copy);
    merged.def("get_type_info", &ov::op::util::MultiSubGraphOp::MergedInputDescription::get_type_info);
    merged.def_readonly("input_index", &ov::op::util::MultiSubGraphOp::MergedInputDescription::m_input_index);
    merged.def_readonly("body_parameter_index",
                        &ov::op::util::MultiSubGraphOp::MergedInputDescription::m_body_parameter_index);
    merged.def_readonly("body_value_index", &ov::op::util::MultiSubGraphOp::MergedInputDescription::m_body_value_index);

    py::class_<ov::op::util::MultiSubGraphOp::InvariantInputDescription,
               std::shared_ptr<ov::op::util::MultiSubGraphOp::InvariantInputDescription>,
               ov::op::util::MultiSubGraphOp::InputDescription>(m, "InvariantInputDescription")
        .def(py::init<>())
        .def(py::init<uint64_t&, uint64_t&>(), py::arg("input_index"), py::arg("body_parameter_index"))
        .def("copy", &ov::op::util::MultiSubGraphOp::InvariantInputDescription::copy)
        .def("get_type_info", &ov::op::util::MultiSubGraphOp::InvariantInputDescription::get_type_info)
        .def_readonly("input_index", &ov::op::util::MultiSubGraphOp::InvariantInputDescription::m_input_index)
        .def_readonly("body_parameter_index",
                      &ov::op::util::MultiSubGraphOp::InvariantInputDescription::m_body_parameter_index);

    py::class_<ov::op::util::MultiSubGraphOp::OutputDescription,
               std::shared_ptr<ov::op::util::MultiSubGraphOp::OutputDescription>,
               PyOutputDescription>
        output(m, "OutputDescription");
    output.doc() =
        "ov::op::util::MultiSubGraphOp::OutputDescription wraps ov::op::util::MultiSubGraphOp::OutputDescription";
    output.def(py::init<>());
    output.def("copy", &ov::op::util::MultiSubGraphOp::OutputDescription::copy);

    py::class_<ov::op::util::MultiSubGraphOp::ConcatOutputDescription,
               std::shared_ptr<ov::op::util::MultiSubGraphOp::ConcatOutputDescription>,
               ov::op::util::MultiSubGraphOp::OutputDescription>(m, "ConcatOutputDescription")
        .def(py::init<>())
        .def(py::init<uint64_t&, uint64_t&, int64_t&, int64_t&, int64_t&, int64_t&, int64_t&>(),
             py::arg("body_value_index"),
             py::arg("output_index"),
             py::arg("start"),
             py::arg("stride"),
             py::arg("part_size"),
             py::arg("end"),
             py::arg("axis"))
        .def("copy", &ov::op::util::MultiSubGraphOp::ConcatOutputDescription::copy)
        .def("get_type_info", &ov::op::util::MultiSubGraphOp::ConcatOutputDescription::get_type_info)
        .def_readonly("output_index", &ov::op::util::MultiSubGraphOp::ConcatOutputDescription::m_output_index)
        .def_readonly("body_value_index", &ov::op::util::MultiSubGraphOp::ConcatOutputDescription::m_body_value_index)
        .def_readonly("start", &ov::op::util::MultiSubGraphOp::ConcatOutputDescription::m_start)
        .def_readonly("stride", &ov::op::util::MultiSubGraphOp::ConcatOutputDescription::m_stride)
        .def_readonly("part_size", &ov::op::util::MultiSubGraphOp::ConcatOutputDescription::m_part_size)
        .def_readonly("end", &ov::op::util::MultiSubGraphOp::ConcatOutputDescription::m_end)
        .def_readonly("axis", &ov::op::util::MultiSubGraphOp::ConcatOutputDescription::m_axis);

    py::class_<ov::op::util::MultiSubGraphOp::BodyOutputDescription,
               std::shared_ptr<ov::op::util::MultiSubGraphOp::BodyOutputDescription>,
               ov::op::util::MultiSubGraphOp::OutputDescription>(m, "BodyOutputDescription")
        .def(py::init<>())
        .def(py::init<uint64_t&, uint64_t&, int64_t&>(),
             py::arg("body_value_index"),
             py::arg("output_index"),
             py::arg("iteration") = -1)
        .def("copy", &ov::op::util::MultiSubGraphOp::BodyOutputDescription::copy)
        .def("get_type_info", &ov::op::util::MultiSubGraphOp::BodyOutputDescription::get_type_info)
        .def_readonly("output_index", &ov::op::util::MultiSubGraphOp::BodyOutputDescription::m_output_index)
        .def_readonly("body_value_index", &ov::op::util::MultiSubGraphOp::BodyOutputDescription::m_body_value_index)
        .def_readonly("iteration", &ov::op::util::MultiSubGraphOp::BodyOutputDescription::m_iteration);
}