// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "permute_nchw_as_nhwc_interpolate.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/opsets/opset1.hpp"

ov::intel_cpu::PermuteNCHWAsNHWCInterpolate::PermuteNCHWAsNHWCInterpolate() {
    auto interpolate_m = ov::pass::pattern::wrap_type<op::v11::Interpolate>();

    matcher_pass_callback callback = [=](pass::pattern::Matcher& m) {
//        auto interpolate = ov::as_type_ptr<op::v11::Interpolate>(m.get_match_root());
//        if (!interpolate) {
//            return false;
//        }
//
//        auto axes = (ov::as_type_ptr<ov::opset1::Constant>(interpolate->input_value(2).get_node_shared_ptr()))
//                        ->cast_vector<int64_t>();
//        if (!(interpolate->get_input_partial_shape(0).size() == 4 && axes[0] == 1 && axes[1] == 2)) {
//            return false;
//        }
//
//        auto input_transpose_sizes = std::make_shared<op::v0::Constant>(element::i64, ov::Shape{4}, Shape{0, 3, 1, 2});
//        auto input_transpose =
//            std::make_shared<ov::opset1::Transpose>(interpolate->input_value(0), input_transpose_sizes);
//
//        std::vector<int64_t> axes_shapes = {2, 3};
//        auto axes_shapes_const =
//            std::make_shared<ov::opset1::Constant>(ov::element::i64, ov::Shape{axes_shapes.size()}, axes_shapes);
//        auto new_interpolate =
//            interpolate->clone_with_new_inputs({input_transpose, interpolate->input_value(1), axes_shapes_const});
//
//        ov::replace_node(interpolate, new_interpolate);
//        new_interpolate->set_friendly_name(interpolate->get_friendly_name());
//        ov::copy_runtime_info(interpolate, new_interpolate);
//
//        auto new_transpose_sizes = std::make_shared<op::v0::Constant>(element::i64, ov::Shape{4}, Shape{0, 2, 3, 1});
//        auto new_transpose = std::make_shared<ov::opset1::Transpose>(new_interpolate->output(0), new_transpose_sizes);
//
//        ov::replace_node(new_interpolate, new_transpose);
//        new_transpose->set_friendly_name(new_interpolate->get_friendly_name());
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(interpolate_m, "PermuteNCHWAsNHWCInterpolate");
    register_matcher(m, callback);
}
