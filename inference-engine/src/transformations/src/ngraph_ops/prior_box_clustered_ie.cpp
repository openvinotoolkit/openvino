// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_ops/prior_box_clustered_ie.hpp"

#include <memory>

#include "ngraph/op/constant.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::PriorBoxClusteredIE::type_info;

op::PriorBoxClusteredIE::PriorBoxClusteredIE(const Output<Node>& input, const Output<Node>& image,
                                             const PriorBoxClusteredAttrs& attrs)
    : Op({input, image}), m_attrs(attrs) {
    constructor_validate_and_infer_types();
}

void op::PriorBoxClusteredIE::validate_and_infer_types() {
    if (get_input_partial_shape(0).is_dynamic() || get_input_partial_shape(1).is_dynamic()) {
        set_output_type(0, element::f32, PartialShape::dynamic(3));
        return;
    }

    auto input_shape = get_input_shape(0);
    auto image_shape = get_input_shape(1);

    size_t num_priors = m_attrs.widths.size();

    set_output_type(0, element::f32, Shape {1, 2, 4 * input_shape[2] * input_shape[3] * num_priors});
}

shared_ptr<Node> op::PriorBoxClusteredIE::copy_with_new_args(const NodeVector& new_args) const {
    check_new_args_count(this, new_args);
    return make_shared<PriorBoxClusteredIE>(new_args.at(0), new_args.at(1), m_attrs);
}
