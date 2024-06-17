// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/sync_tensor.hpp"
#include "openvino/core/type/element_type.hpp"
#include "transformations/rt_info/fused_names_attribute.hpp"

namespace ov {
namespace intel_gpu {
namespace op {

SyncTensor::SyncTensor(const ov::element::Type output_type) : ov::op::Op() {
    m_output_type = output_type;
    validate_and_infer_types();
}

SyncTensor::SyncTensor(const Output<Node>& input,
           const ov::element::Type output_type,
           const TP_MODE tp_mode) : ov::op::Op({input}) {
    m_output_type = output_type;
    m_tp_mode = tp_mode;
    validate_and_infer_types();
}

bool SyncTensor::visit_attributes(ov::AttributeVisitor& visitor) {
    return true;
}

void SyncTensor::validate_and_infer_types() {
    if (get_input_size() > 0) {
        auto output_type = m_output_type == ov::element::undefined ? get_input_element_type(0) : m_output_type;
        set_output_type(0, output_type, get_input_partial_shape(0));
    } else {
        set_output_type(0, m_output_type, ov::PartialShape());
    }
}

std::shared_ptr<Node> SyncTensor::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    switch (new_args.size()) {
    case 0:
        return std::make_shared<SyncTensor>(m_output_type);
    case 1:
        return std::make_shared<SyncTensor>(new_args[0], m_output_type);
    default:
        OPENVINO_THROW("Unable to clone SyncTensor ",
                       this->get_friendly_name(),
                       " Incorrect number of inputs. Expected: 0 or 1. Actual: ",
                       new_args.size());
    }
}

std::vector<ov::PartialShape> shape_infer(const SyncTensor* op, std::vector<ov::PartialShape> input_shapes) {
    std::vector<ov::PartialShape> out_shapes;
    if (op->get_tp_mode() == TP_MODE::ALL_REDUCE) {
        auto out_shape = op->get_input_partial_shape(0);
        out_shapes.push_back(out_shape);
    } else {
        // TBD
    }

    return out_shapes;
}
}  // namespace op
}  // namespace intel_gpu
}  // namespace ov
