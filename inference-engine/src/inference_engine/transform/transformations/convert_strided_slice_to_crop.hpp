//*****************************************************************************
// Copyright 2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.../src/inference_engine/ie_cnn_layer_builder.cpp
//*****************************************************************************

#pragma once

#include <vector>
#include <memory>

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph_ops/crop_ie.hpp>

#include "ngraph/op/experimental/layers/interpolate.hpp"
#include "ngraph/op/experimental/dyn_slice.hpp"

namespace ngraph {
namespace pass {

class ConvertStridedSliceToCrop;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertStridedSliceToCrop: public ngraph::pass::GraphRewrite {
public:
    ConvertStridedSliceToCrop() : GraphRewrite() {
        convert_strided_slice_to_crop();
    }

private:
    void convert_strided_slice_to_crop() {
        auto data = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
        auto m_begin = std::make_shared<pattern::op::Label>(element::i64, Shape{2});
        auto m_end = std::make_shared<pattern::op::Label>(element::i64, Shape{2});
        auto m_stride = std::make_shared<pattern::op::Label>(element::i64, Shape{2});

        auto m_slice = std::make_shared<ngraph::op::DynSlice>(data, m_begin, m_end, m_stride);

        ngraph::graph_rewrite_callback callback = [](pattern::Matcher& m) {
            auto slice = std::dynamic_pointer_cast<ngraph::op::DynSlice> (m.get_match_root());
            if (!slice) {
                return false;
            }

            auto data_node = slice->get_argument(0);
            auto begin_node = std::dynamic_pointer_cast<ngraph::op::Constant>(slice->get_argument(1));
            auto end_node = std::dynamic_pointer_cast<ngraph::op::Constant>(slice->get_argument(2));
            auto stride_node = std::dynamic_pointer_cast<ngraph::op::Constant>(slice->get_argument(3));

            auto partial_input_shape = slice->get_input_partial_shape(0);

            if (!begin_node || !end_node || !stride_node || partial_input_shape.is_dynamic()) {
                return false;
            }

            auto input_shape = slice->get_input_shape(0);

            auto begin = begin_node->get_vector<int64_t>();
            auto end = end_node->get_vector<int64_t>();
            auto stride = stride_node->get_vector<int64_t>();

            bool ones_stride = true;
            for (auto & s : stride) {
                if (s != 1) ones_stride = false;
            }

            if (!ones_stride) return false;

            auto shrink_axis_mask = slice->get_shrink_axis();
            auto new_axis_mask = slice->get_new_axis();
            auto ellipsis_mask = slice->get_ellipsis_mask();
            auto begin_mask = slice->get_lower_bounds_mask();
            auto end_mask = slice->get_upper_bounds_mask();

            if (shrink_axis_mask.size() != 0) {
                std::cout << "StridedSlice: shrink_axis_mask is not supported" << std::endl;
                return false;
            }

            if (new_axis_mask.size() != 0) {
                std::cout << "StridedSlice: new_axis_mask is not supported" << std::endl;
                return false;
            }

            if (ellipsis_mask.size() != 0) {
                std::cout << "StridedSlice: ellipsis_mask is not supported" << std::endl;
                return false;
            }

            std::vector<int64_t> axes(input_shape.size()), offset(input_shape.size()), dim(input_shape.size());

            for (size_t axis = 0; axis < input_shape.size(); ++axis) {
                axes[axis] = axis;

                offset[axis] = begin[axis];
                dim[axis] = end[axis] - begin[axis];

                if (begin_mask.count(axis)) {
                    offset[axis] = 0;
                }
                if (end_mask.count(axis)) {
                    dim[axis] = input_shape[axis] - offset[axis];
                }
            }

            auto crop = std::make_shared<ngraph::op::CropIE> (data_node, axes, dim, offset);
            crop->set_friendly_name(slice->get_friendly_name());
            ngraph::replace_node(m.get_match_root(), std::dynamic_pointer_cast<ngraph::Node>(crop));
            return true;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(m_slice, "ConvertStridedSliceToCrop");
        this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
    }
};
