// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/convert_opset1_to_legacy/convert_strided_slice_to_strided_slice_ie.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset1.hpp>

#include <ngraph_ops/strided_slice_ie.hpp>
#include <ngraph/rt_info.hpp>

void ngraph::pass::ConvertStridedSliceToStridedSliceIE::convert_strided_slice_to_strided_slice_ie() {
    auto slice = std::make_shared<pattern::op::Label>(element::f32, Shape{}, pattern::has_class<opset1::StridedSlice>());

    ngraph::graph_rewrite_callback callback = [](pattern::Matcher& m) {
        auto slice = std::dynamic_pointer_cast<opset1::StridedSlice> (m.get_match_root());
        if (!slice) {
            return false;
        }

        auto data_node = slice->input_value(0);
        auto begin_node = std::dynamic_pointer_cast<opset1::Constant>(slice->input_value(1).get_node_shared_ptr());
        auto end_node = std::dynamic_pointer_cast<opset1::Constant>(slice->input_value(2).get_node_shared_ptr());
        auto stride_node = std::dynamic_pointer_cast<opset1::Constant>(slice->input_value(3).get_node_shared_ptr());

        if (!begin_node || !end_node || !stride_node) {
            return false;
        }

        auto converted_begin = std::make_shared<opset1::Convert>(begin_node, element::i32);
        auto converted_end = std::make_shared<opset1::Convert>(end_node, element::i32);
        auto converted_stride = std::make_shared<opset1::Convert>(stride_node, element::i32);

        auto slice_ie = std::make_shared<ngraph::op::StridedSliceIE>(data_node,
                                                                     converted_begin,
                                                                     converted_end,
                                                                     converted_stride,
                                                                     slice->get_begin_mask(),
                                                                     slice->get_end_mask(),
                                                                     slice->get_new_axis_mask(),
                                                                     slice->get_shrink_axis_mask(),
                                                                     slice->get_ellipsis_mask());
        slice_ie->set_friendly_name(slice->get_friendly_name());

        ngraph::copy_runtime_info(slice, {converted_begin, converted_end, converted_stride, slice_ie});
        ngraph::replace_node(slice, slice_ie);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(slice, "ConvertStridedSliceToStridedSliceIE");
    this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
}