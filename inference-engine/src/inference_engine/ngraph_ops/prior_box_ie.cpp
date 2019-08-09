// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "prior_box_ie.hpp"

#include "ngraph/op/constant.hpp"

using namespace std;
using namespace ngraph;

op::PriorBoxIE::PriorBoxIE(const shared_ptr<Node>& input,
                           const shared_ptr<Node>& image,
                           const PriorBoxAttrs& attrs)
        : Op("PriorBoxIE", check_single_output_args({input, image}))
        , m_attrs(attrs) {
    constructor_validate_and_infer_types();
}

void op::PriorBoxIE::validate_and_infer_types() {
    auto input_shape = get_input_shape(0);
    auto image_shape = get_input_shape(1);

    size_t num_priors = 0;
    // {Prior boxes, Variance-adjusted prior boxes}
    if (m_attrs.scale_all_sizes) {
        num_priors = ((m_attrs.flip ? 2 : 1) * m_attrs.aspect_ratio.size() + 1) *
                     m_attrs.min_size.size() +
                     m_attrs.max_size.size();
    } else {
        num_priors = (m_attrs.flip ? 2 : 1) * m_attrs.aspect_ratio.size() +
                     m_attrs.min_size.size() - 1;
    }

    set_output_type(0, element::f32, Shape{1, 2, 4 * input_shape[2] * input_shape[3] * num_priors});
}

shared_ptr<Node> op::PriorBoxIE::copy_with_new_args(const NodeVector& new_args) const {
    check_new_args_count(this, new_args);
    return make_shared<PriorBoxIE>(new_args.at(0), new_args.at(1), m_attrs);
}
