// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph_ops/interp.hpp>

#include "ngraph/op/experimental/layers/interpolate.hpp"

namespace ngraph {
namespace pass {

class ConvertInterpolateToInterpOrResample;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertInterpolateToInterpOrResample: public ngraph::pass::GraphRewrite {
public:
    ConvertInterpolateToInterpOrResample() : GraphRewrite() {
        convert_interpolate_to_interp_or_resample();
    }

private:
    void convert_interpolate_to_interp_or_resample() {
        auto data = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
        auto shp = std::make_shared<pattern::op::Label>(element::i64, Shape{2});
        auto interpolate = std::make_shared<ngraph::op::Interpolate>(data, shp, ngraph::op::InterpolateAttrs());

        ngraph::graph_rewrite_callback callback = [](pattern::Matcher& m) {
            auto interpolate = std::dynamic_pointer_cast<ngraph::op::Interpolate> (m.get_match_root());

            auto data_node = interpolate->get_argument(0);
            auto out_shape_node = std::dynamic_pointer_cast<ngraph::op::Constant>(interpolate->get_argument(1));
            auto interpolate_attrs = interpolate->get_attrs();

            if (!out_shape_node) {
                return false;
            }

            auto out_spatial_shape = out_shape_node->get_vector<int64_t> ();
            if (out_spatial_shape.size() != 2) {
                return false;
            }

            if (!(interpolate_attrs.axes.size() == 2 && interpolate_attrs.axes.count(2) && interpolate_attrs.axes.count(3))) {
                return false;
            }

            auto attrs = ngraph::op::InterpolateIEAttrs();
            attrs.pad_beg = interpolate_attrs.pads_begin[0];
            attrs.pad_end = interpolate_attrs.pads_end[0];
            attrs.height = out_spatial_shape[0];
            attrs.width = out_spatial_shape[1];
            attrs.align_corners = interpolate_attrs.align_corners;
            attrs.mode = interpolate_attrs.mode;
            attrs.antialias = interpolate_attrs.antialias;

            auto interp = std::make_shared<ngraph::op::Interp> (data_node, attrs);
            interp->set_friendly_name(m.get_match_root()->get_friendly_name());
            ngraph::replace_node(m.get_match_root(), std::dynamic_pointer_cast<ngraph::Node>(interp));
            return true;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(interpolate, "ConvertInterpolateToInterpOrResample");
        this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
    }
};
