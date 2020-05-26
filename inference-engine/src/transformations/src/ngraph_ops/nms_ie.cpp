// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_ops/nms_ie.hpp"

#include <memory>

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::NonMaxSuppressionIE::type_info;

op::NonMaxSuppressionIE::NonMaxSuppressionIE(const Output<Node> &boxes,
                                             const Output<Node> &scores,
                                             const Output<Node> &max_output_boxes_per_class,
                                             const Output<Node> &iou_threshold,
                                             const Output<Node> &score_threshold,
                                             const Shape &output_shape,
                                             int center_point_box,
                                             bool sort_result_descending)
        : Op({boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold}),
          m_center_point_box{center_point_box}, m_sort_result_descending{sort_result_descending}, m_output_shape{output_shape} {
    constructor_validate_and_infer_types();
}


std::shared_ptr<Node> op::NonMaxSuppressionIE::copy_with_new_args(const NodeVector &new_args) const {
    if (new_args.size() != 5) {
        throw ngraph_error("Incorrect number of new arguments");
    }

    return make_shared<NonMaxSuppressionIE>(new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3),
                                            new_args.at(4), m_output_shape, m_center_point_box, m_sort_result_descending);
}

void op::NonMaxSuppressionIE::validate_and_infer_types() {
    set_output_type(0, element::i32, m_output_shape);
}
