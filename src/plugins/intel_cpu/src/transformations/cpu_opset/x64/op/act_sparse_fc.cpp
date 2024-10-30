// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "act_sparse_fc.hpp"

#include "transformations/itt.hpp"
namespace ov {
namespace intel_cpu {

bool ActSparseFCNode::visit_attributes(ov::AttributeVisitor& visitor) {
    INTERNAL_OP_SCOPE(ActSparseFCNode_visit_attributes);
    visitor.start_structure("config");
    visitor.on_attribute("ic", m_config.ic);
    visitor.on_attribute("oc", m_config.oc);
    visitor.on_attribute("threshold", m_config.threshold);
    visitor.finish_structure();
    return true;
}

void ActSparseFCNode::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(ActSparseFCNode_validate_and_infer_types);
    const auto input_size = get_input_size();
    //NODE_VALIDATION_CHECK(this, input_size == expect_input_size);

    const auto& ishape = get_input_partial_shape(0);
    const auto& itype = get_input_element_type(0);

    NODE_VALIDATION_CHECK(this, ishape.rank().is_static() && ishape.rank() == 3, "feature shape rank must be 3");
    const auto feature = ishape[2];
    NODE_VALIDATION_CHECK(this, feature.is_static());
    NODE_VALIDATION_CHECK(this, itype.is_real(), "feature data type must be real");

    auto oshape = ishape;
    oshape[oshape.size() - 1] = m_config.oc;
    set_output_type(0, itype, oshape);
}

std::shared_ptr<Node> ActSparseFCNode::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(ActSparseFCNode_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<ActSparseFCNode>(new_args, m_config);
}
}  // namespace intel_cpu
}  // namespace ov
