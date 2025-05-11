// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/graph/ops/util/multisubgraph.hpp"

#include "openvino/core/node.hpp"
#include "openvino/op/util/sub_graph_base.hpp"
#include "pyopenvino/core/common.hpp"

namespace py = pybind11;

bool MultiSubgraphHelpers::is_constant_or_parameter(const std::shared_ptr<ov::Node>& node) {
    const auto type_name = std::string{node->get_type_name()};
    return type_name == "Constant" || type_name == "Parameter";
}

MultiSubgraphInputDescriptionVector MultiSubgraphHelpers::list_to_input_descriptor(const py::list& inputs) {
    std::vector<ov::op::util::MultiSubGraphOp::InputDescription::Ptr> result;

    for (const auto& in_desc : inputs) {
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

    return result;
}

MultiSubgraphOutputDescriptionVector MultiSubgraphHelpers::list_to_output_descriptor(const py::list& outputs) {
    std::vector<ov::op::util::MultiSubGraphOp::OutputDescription::Ptr> result;

    for (const auto& out_desc : outputs) {
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
               PyInputDescription>(m, "InputDescription")
        .def(py::init<>())
        .def("copy", &ov::op::util::MultiSubGraphOp::InputDescription::copy)
        .def("__repr__", [](const ov::op::util::MultiSubGraphOp::InputDescription& self) {
            return Common::get_simple_repr(self);
        });

    py::class_<ov::op::util::MultiSubGraphOp::SliceInputDescription,
               std::shared_ptr<ov::op::util::MultiSubGraphOp::SliceInputDescription>,
               ov::op::util::MultiSubGraphOp::InputDescription>
        slice(m, "SliceInputDescription");
    slice.doc() = "openvino.impl.op.util.SliceInputDescription wraps ov::op::util::SliceInputDescription";
    slice.def(py::init<>());
    slice.def(py::init<uint64_t, uint64_t, int64_t, int64_t, int64_t, int64_t, int64_t>(),
              py::arg("input_index"),
              py::arg("body_parameter_index"),
              py::arg("start"),
              py::arg("stride"),
              py::arg("part_size"),
              py::arg("end"),
              py::arg("axis"));
    slice.def("copy", &ov::op::util::MultiSubGraphOp::SliceInputDescription::copy);
    slice.def("get_type_info", &ov::op::util::MultiSubGraphOp::SliceInputDescription::get_type_info);
    slice.def_readonly("input_index", &ov::op::util::MultiSubGraphOp::SliceInputDescription::m_input_index);
    slice.def_readonly("body_parameter_index",
                       &ov::op::util::MultiSubGraphOp::SliceInputDescription::m_body_parameter_index);
    slice.def_readonly("start", &ov::op::util::MultiSubGraphOp::SliceInputDescription::m_start);
    slice.def_readonly("stride", &ov::op::util::MultiSubGraphOp::SliceInputDescription::m_stride);
    slice.def_readonly("part_size", &ov::op::util::MultiSubGraphOp::SliceInputDescription::m_part_size);
    slice.def_readonly("end", &ov::op::util::MultiSubGraphOp::SliceInputDescription::m_end);
    slice.def_readonly("axis", &ov::op::util::MultiSubGraphOp::SliceInputDescription::m_axis);
    slice.def("__repr__", [](const ov::op::util::MultiSubGraphOp::SliceInputDescription& self) {
        return Common::get_simple_repr(self);
    });

    py::class_<ov::op::util::MultiSubGraphOp::MergedInputDescription,
               std::shared_ptr<ov::op::util::MultiSubGraphOp::MergedInputDescription>,
               ov::op::util::MultiSubGraphOp::InputDescription>
        merged(m, "MergedInputDescription");
    merged.doc() = "openvino.impl.op.util.MergedInputDescription wraps ov::op::util::MergedInputDescription";
    merged.def(py::init<>());
    merged.def(py::init<uint64_t, uint64_t, uint64_t>(),
               py::arg("input_index"),
               py::arg("body_parameter_index"),
               py::arg("body_value_index"));
    merged.def("copy", &ov::op::util::MultiSubGraphOp::MergedInputDescription::copy);
    merged.def("get_type_info", &ov::op::util::MultiSubGraphOp::MergedInputDescription::get_type_info);
    merged.def_readonly("input_index", &ov::op::util::MultiSubGraphOp::MergedInputDescription::m_input_index);
    merged.def_readonly("body_parameter_index",
                        &ov::op::util::MultiSubGraphOp::MergedInputDescription::m_body_parameter_index);
    merged.def_readonly("body_value_index", &ov::op::util::MultiSubGraphOp::MergedInputDescription::m_body_value_index);
    merged.def("__repr__", [](const ov::op::util::MultiSubGraphOp::MergedInputDescription& self) {
        return Common::get_simple_repr(self);
    });

    py::class_<ov::op::util::MultiSubGraphOp::InvariantInputDescription,
               std::shared_ptr<ov::op::util::MultiSubGraphOp::InvariantInputDescription>,
               ov::op::util::MultiSubGraphOp::InputDescription>
        invariant(m, "InvariantInputDescription");
    invariant.doc() = "openvino.impl.op.util.InvariantInputDescription wraps ov::op::util::InvariantInputDescription";
    invariant.def(py::init<>());
    invariant.def(py::init<uint64_t, uint64_t>(), py::arg("input_index"), py::arg("body_parameter_index"));
    invariant.def("copy", &ov::op::util::MultiSubGraphOp::InvariantInputDescription::copy);
    invariant.def("get_type_info", &ov::op::util::MultiSubGraphOp::InvariantInputDescription::get_type_info);
    invariant.def_readonly("input_index", &ov::op::util::MultiSubGraphOp::InvariantInputDescription::m_input_index);
    invariant.def_readonly("body_parameter_index",
                           &ov::op::util::MultiSubGraphOp::InvariantInputDescription::m_body_parameter_index);
    invariant.def("__repr__", [](const ov::op::util::MultiSubGraphOp::InvariantInputDescription& self) {
        return Common::get_simple_repr(self);
    });

    py::class_<ov::op::util::MultiSubGraphOp::OutputDescription,
               std::shared_ptr<ov::op::util::MultiSubGraphOp::OutputDescription>,
               PyOutputDescription>(m, "OutputDescription")
        .def(py::init<>())
        .def("copy", &ov::op::util::MultiSubGraphOp::OutputDescription::copy)
        .def("__repr__", [](const ov::op::util::MultiSubGraphOp::OutputDescription& self) {
            return Common::get_simple_repr(self);
        });

    py::class_<ov::op::util::MultiSubGraphOp::ConcatOutputDescription,
               std::shared_ptr<ov::op::util::MultiSubGraphOp::ConcatOutputDescription>,
               ov::op::util::MultiSubGraphOp::OutputDescription>
        concat(m, "ConcatOutputDescription");
    concat.doc() = "openvino.impl.op.util.ConcatOutputDescription wraps ov::op::util::ConcatOutputDescription";
    concat.def(py::init<>());
    concat.def(py::init<uint64_t, uint64_t, int64_t, int64_t, int64_t, int64_t, int64_t>(),
               py::arg("body_value_index"),
               py::arg("output_index"),
               py::arg("start"),
               py::arg("stride"),
               py::arg("part_size"),
               py::arg("end"),
               py::arg("axis"));
    concat.def("copy", &ov::op::util::MultiSubGraphOp::ConcatOutputDescription::copy);
    concat.def("get_type_info", &ov::op::util::MultiSubGraphOp::ConcatOutputDescription::get_type_info);
    concat.def_readonly("output_index", &ov::op::util::MultiSubGraphOp::ConcatOutputDescription::m_output_index);
    concat.def_readonly("body_value_index",
                        &ov::op::util::MultiSubGraphOp::ConcatOutputDescription::m_body_value_index);
    concat.def_readonly("start", &ov::op::util::MultiSubGraphOp::ConcatOutputDescription::m_start);
    concat.def_readonly("stride", &ov::op::util::MultiSubGraphOp::ConcatOutputDescription::m_stride);
    concat.def_readonly("part_size", &ov::op::util::MultiSubGraphOp::ConcatOutputDescription::m_part_size);
    concat.def_readonly("end", &ov::op::util::MultiSubGraphOp::ConcatOutputDescription::m_end);
    concat.def_readonly("axis", &ov::op::util::MultiSubGraphOp::ConcatOutputDescription::m_axis);
    concat.def("__repr__", [](const ov::op::util::MultiSubGraphOp::ConcatOutputDescription& self) {
        return Common::get_simple_repr(self);
    });

    py::class_<ov::op::util::MultiSubGraphOp::BodyOutputDescription,
               std::shared_ptr<ov::op::util::MultiSubGraphOp::BodyOutputDescription>,
               ov::op::util::MultiSubGraphOp::OutputDescription>
        body(m, "BodyOutputDescription");
    body.doc() = "openvino.impl.op.util.BodyOutputDescription wraps ov::op::util::BodyOutputDescription";
    body.def(py::init<>());
    body.def(py::init<uint64_t, uint64_t, int64_t>(),
             py::arg("body_value_index"),
             py::arg("output_index"),
             py::arg("iteration") = -1);
    body.def("copy", &ov::op::util::MultiSubGraphOp::BodyOutputDescription::copy);
    body.def("get_type_info", &ov::op::util::MultiSubGraphOp::BodyOutputDescription::get_type_info);
    body.def_readonly("output_index", &ov::op::util::MultiSubGraphOp::BodyOutputDescription::m_output_index);
    body.def_readonly("body_value_index", &ov::op::util::MultiSubGraphOp::BodyOutputDescription::m_body_value_index);
    body.def_readonly("iteration", &ov::op::util::MultiSubGraphOp::BodyOutputDescription::m_iteration);
    body.def("__repr__", [](const ov::op::util::MultiSubGraphOp::BodyOutputDescription& self) {
        return Common::get_simple_repr(self);
    });
}
