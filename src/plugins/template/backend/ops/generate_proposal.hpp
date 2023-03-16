// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "evaluates_map.hpp"

template <ngraph::element::Type_t ET>
bool evaluate(const std::shared_ptr<ngraph::op::v9::GenerateProposals>& op,
              const ngraph::HostTensorVector& outputs,
              const ngraph::HostTensorVector& inputs) {
    const auto& attrs = op->get_attrs();

    if (attrs.post_nms_count < 0) {
        throw ngraph::ngraph_error("The attribute post_nms_count of the operation "
                                   "GenerateProposals must be a "
                                   "nonnegative integer.");
    }

    const auto& output_type = op->get_input_element_type(0);

    const auto& im_info_shape = inputs[0]->get_shape();
    const auto& anchors_shape = inputs[1]->get_shape();
    const auto& deltas_shape = inputs[2]->get_shape();
    const auto& scores_shape = inputs[3]->get_shape();

    const auto im_info_data = get_floats(inputs[0], im_info_shape);
    const auto anchors_data = get_floats(inputs[1], anchors_shape);
    const auto deltas_data = get_floats(inputs[2], deltas_shape);
    const auto scores_data = get_floats(inputs[3], scores_shape);

    std::vector<float> output_rois;
    std::vector<float> output_scores;
    std::vector<int64_t> output_num;

    ngraph::runtime::reference::generate_proposals(im_info_data,
                                                   anchors_data,
                                                   deltas_data,
                                                   scores_data,
                                                   attrs,
                                                   im_info_shape,
                                                   anchors_shape,
                                                   deltas_shape,
                                                   scores_shape,
                                                   output_rois,
                                                   output_scores,
                                                   output_num);

    size_t num_selected = static_cast<size_t>(std::accumulate(output_num.begin(), output_num.end(), 0));

    ngraph::Shape output_rois_shape = ngraph::Shape{num_selected, 4};
    ngraph::Shape output_scores_shape = ngraph::Shape{num_selected};

    outputs[0]->set_element_type(output_type);
    outputs[0]->set_shape(output_rois_shape);
    outputs[1]->set_element_type(output_type);
    outputs[1]->set_shape(output_scores_shape);

    const auto& roi_num_type = op->get_output_element_type(2);
    ngraph::Shape output_roi_num_shape = ngraph::Shape{im_info_shape[0]};
    outputs[2]->set_element_type(roi_num_type);
    outputs[2]->set_shape(output_roi_num_shape);

    ngraph::runtime::reference::generate_proposals_postprocessing(outputs[0]->get_data_ptr(),
                                                                  outputs[1]->get_data_ptr(),
                                                                  outputs[2]->get_data_ptr(),
                                                                  output_type,
                                                                  roi_num_type,
                                                                  output_rois,
                                                                  output_scores,
                                                                  output_num,
                                                                  output_rois_shape,
                                                                  output_scores_shape);

    return true;
}