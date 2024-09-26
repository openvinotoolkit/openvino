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

SyncTensor::SyncTensor(const size_t world_size, const size_t rank, const TP_MODE tp_mode)
    : ov::op::Op(),
      m_world_size(world_size),
      m_rank(rank),
      m_tp_mode(tp_mode) {
    m_gpu_p2p_enabled = true;
    validate_and_infer_types();
}

SyncTensor::SyncTensor(const Output<Node>& input,
                       const size_t world_size,
                       const size_t rank,
                       int split_dimension,
                       const ov::element::Type output_type,
                       const TP_MODE tp_mode)
    : ov::op::Op({input}),
      m_world_size(world_size),
      m_rank(rank),
      m_split_dimension(split_dimension),
      m_output_type(output_type),
      m_tp_mode(tp_mode) {
    m_gpu_p2p_enabled = true;
    set_output_size(1);
    validate_and_infer_types();
}

bool SyncTensor::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("output_type", m_output_type);
    return true;
}

void SyncTensor::validate_and_infer_types_fallback() {
    set_output_size(m_world_size);
    if (get_input_size() > 0) {
        if (m_tp_mode == TP_MODE::ALL_REDUCE) {
            auto original_fc_out = get_input_source_output(0).get_partial_shape();
            std::vector<ov::PartialShape> p_shapes(m_world_size, original_fc_out);
            for (size_t i = 0; i < p_shapes.size(); i++)
                set_output_type(i, m_output_type, p_shapes[i]);
        } else if (m_tp_mode == TP_MODE::ALL_GATHERH) {
            auto split_parts = [](int len, int n) {
                int average = len / n;
                std::vector<int> parts(n, average);
                parts.back() = len - average * (n - 1);
                return parts;
            };
            auto output_type = m_output_type == ov::element::undefined ? get_input_element_type(0) : m_output_type;
            auto input_pshape = get_input_source_output(0).get_partial_shape();
            std::vector<ov::PartialShape> p_shapes(m_world_size, input_pshape);
            auto fc_out_dim_vec = split_parts(m_split_dimension, m_world_size);
            const int64_t axis = ov::util::normalize_axis(-1, input_pshape.size());
            const auto& dimension_at_axis = input_pshape[axis];

            if (dimension_at_axis.is_static()) {
                for (size_t i = 0; i < m_world_size; ++i) {
                    p_shapes[i][axis] = ov::Dimension(fc_out_dim_vec[i]);
                }
            }
            for (size_t i = 0; i < p_shapes.size(); i++)
                set_output_type(i, output_type, p_shapes[i]);
        }
    } else {
        set_output_type(0, m_output_type, ov::PartialShape());
    }
}

void SyncTensor::validate_and_infer_types() {
    const char* env = getenv("OV_GPU_P2P_DISABLED");
    if (env) {
        m_gpu_p2p_enabled = false;
        return validate_and_infer_types_fallback();
    }
    if (get_input_size() > 0) {
        if (m_tp_mode == TP_MODE::ALL_REDUCE) {
            auto original_fc_out = get_input_source_output(0).get_partial_shape();
            std::vector<ov::PartialShape> p_shapes(1, original_fc_out);
            for (size_t i = 0; i < p_shapes.size(); i++)
                set_output_type(i, m_output_type, p_shapes[i]);
        } else if (m_tp_mode == TP_MODE::ALL_GATHERH) {
            set_output_size(2);  // first ouput as result
            auto split_parts = [](int len, int n) {
                int average = len / n;
                std::vector<int> parts(n, average);
                parts.back() = len - average * (n - 1);
                return parts;
            };
            auto output_type = m_output_type == ov::element::undefined ? get_input_element_type(0) : m_output_type;
            auto input_pshape = get_input_source_output(0).get_partial_shape();
            std::vector<ov::PartialShape> p_shapes(2, input_pshape);
            auto fc_out_dim_vec = split_parts(m_split_dimension, m_world_size);
            const int64_t axis = ov::util::normalize_axis(-1, input_pshape.size());
            const auto& dimension_at_axis = input_pshape[axis];

            if (dimension_at_axis.is_static()) {
                p_shapes[0][axis] = ov::Dimension(m_split_dimension);
                p_shapes[1][axis] = ov::Dimension(fc_out_dim_vec[m_rank]);
            }
            for (size_t i = 0; i < p_shapes.size(); i++) {
                // std::cout << "SyncTensor::validate_and_infer_types: shape[" << i << "] = " << p_shapes[i].to_string()
                //           << std::endl;
                set_output_type(i, output_type, p_shapes[i]);
            }
        }
    } else {
        set_output_type(0, m_output_type, ov::PartialShape());
    }
}

std::shared_ptr<Node> SyncTensor::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    OPENVINO_ASSERT(new_args.size() == 1,
                    "Unable to clone SyncTensor with name ",
                    this->get_friendly_name(),
                    ", which should only has 1 input!");
    return std::make_shared<SyncTensor>(new_args[0], m_world_size, m_rank, m_split_dimension, m_output_type, m_tp_mode);
}

std::vector<ov::PartialShape> shape_infer(const SyncTensor* op, std::vector<ov::PartialShape> input_shapes) {
    std::vector<ov::PartialShape> out_shapes;
    for (size_t i = 0; i < op->get_output_size(); i++) {
        out_shapes.push_back(input_shapes[0]);
    }
    if (!op->get_gpu_p2p_enabled()) {
        return out_shapes;
    }
    auto first_shape = input_shapes[0];
    for (size_t i = 1; i < op->get_world_size(); i++) {
        // Accumulate inputs to output, which is in the first output.
        // TODO: fix unevenly split, such as [1024, 1024, 1024, 1022]
        if (op->get_tp_mode() == TP_MODE::ALL_GATHERH)
            first_shape[-1] += input_shapes[0][-1];
        else if (op->get_tp_mode() == TP_MODE::ALL_GATHERV)
            first_shape[1] += input_shapes[0][1];  // need fixed.
    }
    out_shapes[0] = first_shape;

    // for (size_t i = 0; i < out_shapes.size(); i++) {
    //     std::cout << "SyncTensor - shape_infer[" << i << "] = " << out_shapes[i].to_string() << std::endl;
    // }
    return out_shapes;
}
}  // namespace op
}  // namespace intel_gpu
}  // namespace ov
