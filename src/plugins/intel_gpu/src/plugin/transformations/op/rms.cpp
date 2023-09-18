// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/rms.hpp"

namespace ov {
namespace intel_gpu {
namespace op {

RMS::RMS(const Output<Node>& data,
         const Output<Node>& gamma,
         double epsilson)
    : Op({data, gamma}), m_epsilon(epsilson) {
    validate_and_infer_types();
}

bool RMS::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("epsilon", m_epsilon);
    return true;
}

void RMS::validate_and_infer_types() {
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

std::shared_ptr<Node> RMS::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<RMS>(new_args.at(0),
                                 new_args.at(1),
                                 m_epsilon);
}

}  // namespace op
}  // namespace intel_gpu
}  // namespace ov
