// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph_ops/strided_slice_ie.hpp>

#include "ngraph/op/strided_slice.hpp"
#include "ngraph/op/convert.hpp"

namespace ngraph {
namespace pass {

class ConvertStridedSliceToStridedSliceIE;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertStridedSliceToStridedSliceIE: public ngraph::pass::GraphRewrite {
public:
    ConvertStridedSliceToStridedSliceIE() : GraphRewrite() {
        convert_strided_slice_to_strided_slice_ie();
    }

private:
    void convert_strided_slice_to_strided_slice_ie() {
        auto data = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
        auto m_begin = std::make_shared<pattern::op::Label>(element::i64, Shape{2});
        auto m_end = std::make_shared<pattern::op::Label>(element::i64, Shape{2});
        auto m_stride = std::make_shared<pattern::op::Label>(element::i64, Shape{2});
        std::vector<int64_t> begin_mask = {0, 0, 0, 0};
        std::vector<int64_t> end_mask = {0, 0, 0, 0};
        auto m_slice = std::make_shared<ngraph::op::v1::StridedSlice>(data, m_begin, m_end, m_stride, begin_mask, end_mask);

        ngraph::graph_rewrite_callback callback = [](pattern::Matcher& m) {
            auto strided_slice = std::dynamic_pointer_cast<ngraph::op::v1::StridedSlice> (m.get_match_root());
            if (!strided_slice) {
                return false;
            }

            auto data_node = strided_slice->get_argument(0);
            auto begin_node = std::dynamic_pointer_cast<ngraph::op::Constant>(strided_slice->get_argument(1));
            auto end_node = std::dynamic_pointer_cast<ngraph::op::Constant>(strided_slice->get_argument(2));
            auto stride_node = std::dynamic_pointer_cast<ngraph::op::Constant>(strided_slice->get_argument(3));

            auto output_shape = strided_slice->get_output_shape(0);

            if (!begin_node || !end_node || !stride_node) {
                return false;
            }

            auto shrink_axis_mask = strided_slice->get_shrink_axis_mask();
            auto new_axis_mask = strided_slice->get_new_axis_mask();
            auto ellipsis_mask = strided_slice->get_ellipsis_mask();
            auto begin_mask = strided_slice->get_begin_mask();
            auto end_mask = strided_slice->get_end_mask();

            auto converted_begin = std::make_shared<ngraph::op::Convert>(begin_node, ngraph::element::Type_t::i32);
            auto converted_end = std::make_shared<ngraph::op::Convert>(end_node, ngraph::element::Type_t::i32);
            auto converted_stride = std::make_shared<ngraph::op::Convert>(stride_node, ngraph::element::Type_t::i32);

            auto strided_slice_ie = std::make_shared<ngraph::op::StridedSliceIE>(data_node, converted_begin,
                    converted_end, converted_stride, begin_mask, end_mask, new_axis_mask, shrink_axis_mask,
                    ellipsis_mask, output_shape);
            strided_slice_ie->set_friendly_name(strided_slice->get_friendly_name());

            ngraph::replace_node(m.get_match_root(), strided_slice_ie);
            return true;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(m_slice, "ConvertStridedSliceToStridedSliceIE");
        this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
    }
};
