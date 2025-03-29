// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_ops/generate_proposals_ie_internal.hpp"

#include <memory>

#include "itt.hpp"

using namespace std;
using namespace ov;

op::internal::GenerateProposalsIEInternal::GenerateProposalsIEInternal(const Output<Node>& im_info,
                                                                       const Output<Node>& anchors,
                                                                       const Output<Node>& deltas,
                                                                       const Output<Node>& scores,
                                                                       const Attributes& attrs,
                                                                       const element::Type& roi_num_type)
    : Base(im_info, anchors, deltas, scores, attrs, roi_num_type) {
    validate_and_infer_types();
}

std::shared_ptr<Node> op::internal::GenerateProposalsIEInternal::clone_with_new_inputs(
    const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(internal_GenerateProposalsIEInternal_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<op::internal::GenerateProposalsIEInternal>(new_args.at(0),
                                                                  new_args.at(1),
                                                                  new_args.at(2),
                                                                  new_args.at(3),
                                                                  get_attrs(),
                                                                  get_roi_num_type());
}

void op::internal::GenerateProposalsIEInternal::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(internal_GenerateProposalsIEInternal_validate_and_infer_types);
    Base::validate_and_infer_types();

    const auto im_info_shape = get_input_partial_shape(0);
    const auto num_batches = im_info_shape[0];
    NODE_VALIDATION_CHECK(this, num_batches.is_static(), "Number of batches must be static");

    const Dimension post_nms_count{get_attrs().post_nms_count};
    const auto first_dim_shape = num_batches * post_nms_count;

    const auto rois_shape = ov::PartialShape({first_dim_shape, 4});
    const auto scores_shape = ov::PartialShape({first_dim_shape});
    const auto roisnum_shape = ov::PartialShape({num_batches});

    const auto input_type = get_input_element_type(0);
    set_output_type(0, input_type, rois_shape);
    set_output_type(1, input_type, scores_shape);
    set_output_type(2, get_roi_num_type(), roisnum_shape);
}
