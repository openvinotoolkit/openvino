// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/sync_tensor.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/validation_util.hpp"
#include "transformations/rt_info/fused_names_attribute.hpp"

namespace ov {
namespace intel_gpu {
namespace op {

SyncTensor::SyncTensor(const ov::element::Type output_type) : ov::op::Op() {
    set_output_size(1);
    m_output_type = output_type;
    validate_and_infer_types();
}

SyncTensor::SyncTensor(const size_t world_size, const size_t split_dimenstion) : ov::op::Op() {
    m_world_size = world_size;
    m_split_dimension = split_dimenstion;
    validate_and_infer_types();
}

SyncTensor::SyncTensor(const Output<Node>& input,
            const size_t world_size,
            const size_t split_dimension,
            const ov::element::Type output_type,
            const TP_MODE tp_mode) : ov::op::Op({input}),
            m_world_size(world_size),
            m_split_dimension(split_dimension) {
    set_output_size(1); // 2 outputs for now
    m_output_type = output_type;
    m_tp_mode = tp_mode;
    validate_and_infer_types();
}

bool SyncTensor::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("output_type", m_output_type);
    return true;
}

void SyncTensor::validate_and_infer_types() {
    if (get_input_size() > 0) {
        auto output_type = m_output_type == ov::element::undefined ? get_input_element_type(0) : m_output_type;
        auto input_pshape = get_input_source_output(0).get_partial_shape();
        ov::PartialShape p_shapes = input_pshape;
        const int64_t axis = ov::util::normalize_axis("get split axis", -1, input_pshape.rank());

        p_shapes[axis] = m_split_dimension;
        set_output_type(0, output_type, p_shapes);
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
        return std::make_shared<SyncTensor>(new_args[0], m_world_size, m_split_dimension, m_output_type);
    default:
        OPENVINO_THROW("Unable to clone SyncTensor ",
                       this->get_friendly_name(),
                       " Incorrect number of inputs. Expected: 0 or 1. Actual: ",
                       new_args.size());
    }
}

std::vector<ov::PartialShape> shape_infer(const SyncTensor* op, std::vector<ov::PartialShape> input_shapes) {
    std::vector<ov::PartialShape> out_shapes;
    size_t split_dimenstion = op->get_split_dim();
    if (op->get_tp_mode() == TP_MODE::ALL_REDUCE) {
        auto out_shape = op->get_input_partial_shape(0);
        for (size_t i = 0; i < op->get_output_size(); i++)
            out_shapes.push_back(out_shape);
    } else if (op->get_tp_mode() == TP_MODE::ALL_GATHERH) {
        ov::PartialShape out_shape = input_shapes[0];
        const int64_t axis = ov::util::normalize_axis("get split axis", -1, input_shapes[0].rank());
        out_shape[axis] = split_dimenstion;
        out_shapes.push_back(out_shape);
    }
    return out_shapes;
}
}  // namespace op
}  // namespace intel_gpu
}  // namespace ov
