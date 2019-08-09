// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "prior_box_clustered_ie.hpp"

#include "ngraph/op/constant.hpp"

using namespace std;
using namespace ngraph;

op::PriorBoxClusteredIE::PriorBoxClusteredIE(const shared_ptr<Node>& input,
                           const shared_ptr<Node>& image,
                           const PriorBoxClusteredAttrs& attrs)
        : Op("PriorBoxClusteredIE", check_single_output_args({input, image}))
        , m_attrs(attrs) {
    constructor_validate_and_infer_types();
}

void op::PriorBoxClusteredIE::validate_and_infer_types() {
    auto input_shape = get_input_shape(0);
    auto image_shape = get_input_shape(1);

    size_t num_priors = m_attrs.num_priors;

    set_output_type(0, element::f32, Shape{1, 2, 4 * input_shape[2] * input_shape[3] * num_priors});
}

shared_ptr<Node> op::PriorBoxClusteredIE::copy_with_new_args(const NodeVector& new_args) const {
    check_new_args_count(this, new_args);
    return make_shared<PriorBoxClusteredIE>(new_args.at(0), new_args.at(1), m_attrs);
}

