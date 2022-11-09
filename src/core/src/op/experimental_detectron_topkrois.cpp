// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/experimental_detectron_topkrois.hpp"

#include <experimental_detectron_topkrois_shape_inference.hpp>

#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

BWDCMP_RTTI_DEFINITION(op::v6::ExperimentalDetectronTopKROIs);

op::v6::ExperimentalDetectronTopKROIs::ExperimentalDetectronTopKROIs(const Output<Node>& input_rois,
                                                                     const Output<Node>& rois_probs,
                                                                     size_t max_rois)
    : Op({input_rois, rois_probs}),
      m_max_rois(max_rois) {
    constructor_validate_and_infer_types();
}

bool op::v6::ExperimentalDetectronTopKROIs::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v6_ExperimentalDetectronTopKROIs_visit_attributes);
    visitor.on_attribute("max_rois", m_max_rois);
    return true;
}

shared_ptr<Node> op::v6::ExperimentalDetectronTopKROIs::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v6_ExperimentalDetectronTopKROIs_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<op::v6::ExperimentalDetectronTopKROIs>(new_args.at(0), new_args.at(1), m_max_rois);
}

void op::v6::ExperimentalDetectronTopKROIs::validate_and_infer_types() {
    OV_OP_SCOPE(v6_ExperimentalDetectronTopKROIs_validate_and_infer_types);
    const auto input_rois_shape = get_input_partial_shape(0);
    const auto rois_probs_shape = get_input_partial_shape(1);

    std::vector<ov::PartialShape> output_shapes = {ov::PartialShape{}};
    std::vector<ov::PartialShape> input_shapes = {input_rois_shape, rois_probs_shape};
    shape_infer(this, input_shapes, output_shapes);
    set_output_type(0, get_input_element_type(0), output_shapes[0]);
}
