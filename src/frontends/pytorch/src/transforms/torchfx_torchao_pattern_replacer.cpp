// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "torchfx_torchao_pattern_replacer.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "utils.hpp"
#include "utils_quantize.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

using namespace ov::op;
using namespace ov::pass::pattern;

WeightINT4PackMMReplacer::WeightINT4PackMMReplacer() {
    const auto& const_1 = wrap_type<v0::Constant>();
    const auto& const_2 = wrap_type<v0::Constant>();
    const auto& const_3 = wrap_type<v0::Constant>();

    const auto& weight_int4pack_mm = wrap_type<ov::op::util::FrameworkNode>({any_input(), const_1, const_2, const_3});

    ov::matcher_pass_callback callback = [=](Matcher& m) {
        auto weight_int4pack_mm = m.get_match_root();
        if (!weight_int4pack_mm) {
            return false;
        }
        const auto& pattern_map = m.get_pattern_value_map();
        if (!(weight_int4pack_mm = cast_fw_node(m.get_match_root(), "aten._weight_int4pack_mm.default"))) {
            return false;
        }
        auto wt_const = std::dynamic_pointer_cast<v0::Constant>(weight_int4pack_mm->get_input_node_shared_ptr(1));
        auto wt_const_flat = std::make_shared<v0::Constant>(wt_const->get_element_type(),
                                                            Shape({shape_size(wt_const->get_shape()), 1}),
                                                            wt_const->get_data_ptr<uint8_t>());
        std::vector<uint64_t> broadcast_shape_vec(wt_const_flat->get_shape());
        broadcast_shape_vec[1] = 8;
        auto broadcast_shape_const = std::make_shared<v0::Constant>(element::i32, Shape({2}), broadcast_shape_vec);

        auto broadcast = std::make_shared<v3::Broadcast>(wt_const_flat, broadcast_shape_const);
        OutputVector outputs_broadcast(broadcast->get_output_size());
        OutputVector broadcast_inputs(2);
        broadcast_inputs[0] = wt_const_flat->outputs()[0];
        broadcast_inputs[1] = broadcast_shape_const->outputs()[0];
        if (!broadcast->constant_fold(outputs_broadcast, broadcast_inputs)) {
            return false;
        }

        auto broadcast_out_const = std::dynamic_pointer_cast<v0::Constant>(outputs_broadcast[0].get_node_shared_ptr());
        auto broadcast_out_ptr =
            const_cast<int32_t*>(reinterpret_cast<const int32_t*>(broadcast_out_const->get_data_ptr()));
        for (size_t k = 0; k < shape_size(outputs_broadcast[0].get_shape()); ++k) {
            int32_t shift_val = (k % 8) * 4;
            broadcast_out_ptr[k] = (broadcast_out_ptr[k] >> shift_val) & 15;
        }
        std::vector<uint64_t> wt_ordered_shape(2);
        wt_ordered_shape[0] = wt_const->get_shape()[0] * 8;
        wt_ordered_shape[1] = wt_const->get_shape()[1] * wt_const->get_shape()[2] * wt_const->get_shape()[3];
        auto wt_const_ordered = std::make_shared<v0::Constant>(wt_const->get_element_type(), Shape(wt_ordered_shape));
        auto wt_ordered_ptr = const_cast<int32_t*>(reinterpret_cast<const int32_t*>(wt_const_ordered->get_data_ptr()));
        for (uint64_t b = 0; b < wt_ordered_shape[0] / 64; b++) {
            for (uint64_t j = 0; j < wt_ordered_shape[1]; j++) {
                for (uint64_t i = 0; i < 32; i++) {
                    uint64_t l = 0;
                    uint64_t m = (i * 2);
                    l = b * 64 * (broadcast_out_const->get_shape()[0] / wt_ordered_shape[0]) + j * 8 +
                        (m / broadcast_out_const->get_shape()[1]);
                    m = m % broadcast_out_const->get_shape()[1];
                    wt_ordered_ptr[(b * 64 + i) * wt_ordered_shape[1] + j] = broadcast_out_ptr[l * 8 + m];
                    wt_ordered_ptr[(b * 64 + i + 32) * wt_ordered_shape[1] + j] = broadcast_out_ptr[l * 8 + m + 1];
                }
            }
        }

        auto transpose_order = v0::Constant::create(element::i32, Shape{2}, {1, 0});
        auto transpose = std::make_shared<v1::Transpose>(wt_const_ordered, transpose_order);
        OutputVector outputs_transpose(transpose->get_output_size());
        OutputVector transpose_inputs(2);
        transpose_inputs[0] = wt_const_ordered->outputs()[0];
        transpose_inputs[1] = transpose_order->outputs()[0];
        if (!transpose->constant_fold(outputs_transpose, transpose_inputs)) {
            return false;
        }
        auto g_const = std::dynamic_pointer_cast<v0::Constant>(weight_int4pack_mm->get_input_node_shared_ptr(2));
        auto g_ptr = const_cast<int64_t*>(reinterpret_cast<const int64_t*>(g_const->get_data_ptr()));
        uint64_t g = (uint64_t)(g_ptr[0]);
        if (g > outputs_transpose[0].get_shape()[1]) {
            g = outputs_transpose[0].get_shape()[1];
        }
        auto wt_i32 = std::make_shared<v0::Constant>(
            outputs_transpose[0].get_node_shared_ptr()->get_element_type(),
            Shape({outputs_transpose[0].get_node_shared_ptr()->get_shape()[0] / g,
                   g,
                   outputs_transpose[0].get_node_shared_ptr()->get_shape()[1]}),
            std::dynamic_pointer_cast<v0::Constant>(outputs_transpose[0].get_node_shared_ptr())
                ->get_data_ptr<uint8_t>());

        auto wt_i32_ptr = const_cast<int32_t*>(reinterpret_cast<const int32_t*>(wt_i32->get_data_ptr()));

        auto convert_to_u4 = std::make_shared<v0::Convert>(wt_i32, element::u4);
        OutputVector outputs_to_u4(convert_to_u4->get_output_size());
        if (!convert_to_u4->constant_fold(outputs_to_u4, wt_i32->outputs())) {
            return false;
        }

        auto sz_const = std::dynamic_pointer_cast<v0::Constant>(weight_int4pack_mm->get_input_node_shared_ptr(3));
        const auto two = v0::Constant::create(element::i32, Shape{}, {2});
        const auto split = std::make_shared<v1::Split>(sz_const, two, 2);
        OutputVector outputs_split(split->get_output_size());
        OutputVector split_inputs(2);
        split_inputs[0] = sz_const->outputs()[0];
        split_inputs[1] = two->outputs()[0];
        if (!split->constant_fold(outputs_split, split_inputs)) {
            return false;
        }

        const auto divide = std::make_shared<v1::Divide>(outputs_split[1], outputs_split[0]);
        OutputVector outputs_divide(divide->get_output_size());
        OutputVector divide_inputs(2);
        divide_inputs[0] = outputs_split[1];
        divide_inputs[1] = outputs_split[0];
        if (!divide->constant_fold(outputs_divide, divide_inputs)) {
            return false;
        }

        const auto eight = v0::Constant::create(sz_const->get_element_type(), Shape{}, {8.0});
        const auto subtract = std::make_shared<v1::Subtract>(eight, outputs_divide[0]);
        OutputVector outputs_subtract(subtract->get_output_size());
        OutputVector subtract_inputs(2);
        subtract_inputs[0] = eight->outputs()[0];
        subtract_inputs[1] = outputs_divide[0];
        if (!subtract->constant_fold(outputs_subtract, subtract_inputs)) {
            return false;
        }

        auto new_scales = std::make_shared<v0::Constant>(
            outputs_split[0].get_element_type(),
            Shape({outputs_split[0].get_shape()[0], 1, outputs_split[0].get_shape()[1]}),
            std::dynamic_pointer_cast<v0::Constant>(outputs_split[0].get_node_shared_ptr())->get_data_ptr<uint8_t>());
        auto new_zeros = std::make_shared<v0::Constant>(
            outputs_subtract[0].get_element_type(),
            Shape({outputs_subtract[0].get_shape()[0], 1, outputs_subtract[0].get_shape()[1]}),
            std::dynamic_pointer_cast<v0::Constant>(outputs_subtract[0].get_node_shared_ptr())
                ->get_data_ptr<uint8_t>());

        auto convert_scales_to_float = std::make_shared<v0::Convert>(new_scales, element::f32);
        OutputVector outputs_scales_to_float(convert_scales_to_float->get_output_size());
        if (!convert_scales_to_float->constant_fold(outputs_scales_to_float, new_scales->outputs())) {
            return false;
        }
        auto new_scales_ptr = const_cast<float*>(reinterpret_cast<const float*>(
            std::dynamic_pointer_cast<v0::Constant>(outputs_scales_to_float[0].get_node_shared_ptr())->get_data_ptr()));

        auto convert_zeros_to_float = std::make_shared<v0::Convert>(new_zeros, element::f32);
        OutputVector outputs_zeros_to_float(convert_zeros_to_float->get_output_size());
        if (!convert_zeros_to_float->constant_fold(outputs_zeros_to_float, new_zeros->outputs())) {
            return false;
        }
        auto new_zeros_ptr = const_cast<float*>(reinterpret_cast<const float*>(
            std::dynamic_pointer_cast<v0::Constant>(outputs_zeros_to_float[0].get_node_shared_ptr())->get_data_ptr()));

        auto new_convert =
            std::make_shared<v0::Convert>(outputs_to_u4[0].get_node_shared_ptr(), new_zeros->get_element_type());
        auto new_subtract = std::make_shared<v1::Subtract>(new_convert, new_zeros);
        auto new_mult = std::make_shared<v1::Multiply>(new_subtract, new_scales);
        auto new_shape = v0::Constant::create(element::i32,
                                              Shape{outputs_transpose[0].get_shape().size()},
                                              outputs_transpose[0].get_shape());
        auto new_reshape = std::make_shared<v1::Reshape>(new_mult, new_shape, false);
        auto new_matmul = std::make_shared<v0::MatMul>(weight_int4pack_mm->get_input_node_shared_ptr(0), new_reshape);

        replace_node(weight_int4pack_mm, new_matmul);

        return true;
    };

    auto m = std::make_shared<Matcher>(weight_int4pack_mm, "ov::frontend::pytorch::pass::WeightINT4PackMMReplacer");
    this->register_matcher(m, callback);
};

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
