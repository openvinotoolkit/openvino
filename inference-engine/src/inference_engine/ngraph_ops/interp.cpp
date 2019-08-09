// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "interp.hpp"

#include <limits>
#include <memory>

#include "ngraph/op/constant.hpp"

using namespace std;
using namespace ngraph;

op::Interp::Interp(const std::shared_ptr<Node>& image, const InterpolateIEAttrs& attrs)
        : Op("Interp", check_single_output_args({image}))
        , m_attrs(attrs) {
    constructor_validate_and_infer_types();
}

void op::Interp::validate_and_infer_types() {
    if (get_input_partial_shape(0).is_static()) {
        Shape input_shape = get_input_partial_shape(0).to_shape();
        Shape output_shape(4);
        // Assumes {N, C, H, W}
        output_shape[0] = input_shape[0];
        output_shape[1] = input_shape[1];

        auto is_zero = [](float value) {
            return std::fabs(value) < std::numeric_limits<float>::epsilon();
        };

        bool should_scale = !(is_zero(m_attrs.zoom_factor) && is_zero(m_attrs.shrink_factor) &&
                              is_zero(m_attrs.scale_factor));

        if (should_scale) {
            float scale = m_attrs.scale_factor;
            if (!is_zero(m_attrs.shrink_factor) || !is_zero(m_attrs.zoom_factor)) {
                if (!is_zero(m_attrs.zoom_factor)) {
                    scale = m_attrs.zoom_factor;
                }
                if (!is_zero(m_attrs.shrink_factor)) {
                    scale /= m_attrs.shrink_factor;
                }
            }
            output_shape[2] = input_shape[2] * scale;
            output_shape[3] = input_shape[3] * scale;
        }

        if (m_attrs.height > 0) {
            output_shape[2] = m_attrs.height;
        }
        if (m_attrs.width > 0) {
            output_shape[3] = m_attrs.width;
        }

        set_output_type(0, get_input_element_type(0), output_shape);
    } else {
        set_output_type(0, get_input_element_type(0), PartialShape::dynamic());
    }
}

shared_ptr<Node> op::Interp::copy_with_new_args(const NodeVector& new_args) const {
    check_new_args_count(this, new_args);
    return make_shared<Interp>(new_args.at(0), m_attrs);
}

op::ResampleV2::ResampleV2(const std::shared_ptr<Node>& image,
                                   const std::shared_ptr<Node>& output_shape,
                                   const InterpolateIEAttrs& attrs)
        : Op("ResampleV2", check_single_output_args({image, output_shape}))
        , m_attrs(attrs) {
    constructor_validate_and_infer_types();
}

void op::ResampleV2::validate_and_infer_types() {
    set_input_is_relevant_to_shape(1);
    if (auto const_shape = dynamic_pointer_cast<op::Constant>(get_argument(1))) {
        NODE_VALIDATION_CHECK(this,
                              shape_size(const_shape->get_shape()) == 4,
                              "Layer shape must have rank 4",
                              const_shape->get_shape());

        auto out_shape = static_cast<const int64_t*>(const_shape->get_data_ptr());
        Shape output_shape;
        for (size_t i = 0; i < 4; i++) {
            output_shape.push_back((out_shape[i] > 0) ? out_shape[i] : 0);
        }
        set_output_type(0, get_input_element_type(0), output_shape);
    } else {
        set_output_type(0, get_input_element_type(0), PartialShape::dynamic());
    }
}

shared_ptr<Node> op::ResampleV2::copy_with_new_args(const NodeVector& new_args) const {
    check_new_args_count(this, new_args);
    return make_shared<ResampleV2>(new_args.at(0), new_args.at(1), m_attrs);
}