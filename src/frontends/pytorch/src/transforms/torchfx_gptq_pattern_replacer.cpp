// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "torchfx_gptq_pattern_replacer.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/bitwise_and.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/unsqueeze.hpp"
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

uint32_t read_u4_data(const void* array, size_t index) {
    auto arr_u32 = reinterpret_cast<const uint32_t*>(array);
    size_t idx_u32 = index / 8;
    size_t offset_u32 = index % 8;
    uint32_t val = arr_u32[idx_u32];
    val = val >> (offset_u32 * 4);
    val = val & 15;
    return val;
};

GPTQDecompressionReplacer::GPTQDecompressionReplacer() {
    const auto& const_1 = wrap_type<v0::Constant>();
    const auto& const_2 = wrap_type<v0::Constant>();
    const auto& unsqueeze_1 = wrap_type<v0::Unsqueeze>({const_1, const_2});
    const auto& const_abs = wrap_type<v0::Constant>();
    const auto& abs = wrap_type<v0::Abs>({const_abs});
    const auto& broadcast = wrap_type<ov::op::v3::Broadcast>({unsqueeze_1, abs});
    const auto& const_3 = wrap_type<v0::Constant>();
    const auto& const_4 = wrap_type<v0::Constant>();
    const auto& unsqueeze_2 = wrap_type<v0::Unsqueeze>({const_3, const_4});
    const auto& bitwise_right_shift = wrap_type<ov::op::util::FrameworkNode>({broadcast, unsqueeze_2});
    const auto& convert_1 = wrap_type<v0::Convert>({bitwise_right_shift});
    const auto& const_5 = wrap_type<v0::Constant>();
    const auto& convert_3 = wrap_type<v0::Convert>({bitwise_right_shift});
    const auto& convert_4 = wrap_type<v0::Convert>({const_5});
    const auto& add = wrap_type<v1::Add>({convert_3, convert_4});
    const auto& add_or_convert = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{add, convert_1});
    const auto& const_6 = wrap_type<v0::Constant>();
    const auto& convert_2 = wrap_type<v0::Convert>({const_6});
    const auto& bitwise_and = wrap_type<ov::op::v13::BitwiseAnd>({add_or_convert, convert_2});

    ov::matcher_pass_callback callback = [=](Matcher& m) {
        auto bitwise_and = m.get_match_root();
        if (!bitwise_and) {
            return false;
        }
        const auto& pattern_map = m.get_pattern_value_map();
        auto unsqueeze_1_node = pattern_map.at(unsqueeze_1).get_node_shared_ptr();
        auto unsqueeze_1_in0_const =
            std::dynamic_pointer_cast<v0::Constant>(unsqueeze_1_node->get_input_node_shared_ptr(0));
        auto unsqueeze_1_in1_const =
            std::dynamic_pointer_cast<v0::Constant>(unsqueeze_1_node->get_input_node_shared_ptr(1));
        auto abs_node = pattern_map.at(abs).get_node_shared_ptr();
        auto abs_in_const = std::dynamic_pointer_cast<v0::Constant>(abs_node->get_input_node_shared_ptr(0));
        auto broadcast_node = pattern_map.at(broadcast).get_node_shared_ptr();
        auto unsqueeze_2_node = pattern_map.at(unsqueeze_2).get_node_shared_ptr();
        auto unsqueeze_2_in0_const =
            std::dynamic_pointer_cast<v0::Constant>(unsqueeze_2_node->get_input_node_shared_ptr(0));
        auto unsqueeze_2_in1_const =
            std::dynamic_pointer_cast<v0::Constant>(unsqueeze_2_node->get_input_node_shared_ptr(1));

        OutputVector outputs_1(unsqueeze_1_node->get_output_size());
        OutputVector unsqueeze_1_inputs(2);
        unsqueeze_1_inputs[0] = unsqueeze_1_in0_const->outputs()[0];
        unsqueeze_1_inputs[1] = unsqueeze_1_in1_const->outputs()[0];
        if (!unsqueeze_1_node->constant_fold(outputs_1, unsqueeze_1_inputs)) {
            return false;
        }

        OutputVector outputs_2(abs_node->get_output_size());
        if (!abs_node->constant_fold(outputs_2, abs_in_const->outputs())) {
            return false;
        }

        OutputVector outputs_3(broadcast_node->get_output_size());
        OutputVector broadcast_inputs(2);
        broadcast_inputs[0] = outputs_1[0];
        broadcast_inputs[1] = outputs_2[0];
        if (!broadcast_node->constant_fold(outputs_3, broadcast_inputs)) {
            return false;
        }

        OutputVector outputs_4(unsqueeze_2_node->get_output_size());
        OutputVector unsqueeze_2_inputs(2);
        unsqueeze_2_inputs[0] = unsqueeze_2_in0_const->outputs()[0];
        unsqueeze_2_inputs[1] = unsqueeze_2_in1_const->outputs()[0];
        if (!unsqueeze_2_node->constant_fold(outputs_4, unsqueeze_2_inputs)) {
            return false;
        }
        const int32_t* rs_in0 =
            std::dynamic_pointer_cast<v0::Constant>(outputs_3[0].get_node_shared_ptr())->get_data_ptr<int32_t>();
        const int32_t* rs_in1 =
            std::dynamic_pointer_cast<v0::Constant>(outputs_4[0].get_node_shared_ptr())->get_data_ptr<int32_t>();
        auto shifted_const = std::make_shared<v0::Constant>(element::i32, outputs_3[0].get_shape());
        auto dst = const_cast<int32_t*>(reinterpret_cast<const int32_t*>(shifted_const->get_data_ptr()));
        if (!dst)
            return false;

        // TODO: Bitwise right shift operation below might need to be
        // optimized to reduce FIL.
        size_t rs_in0_shape_size = shape_size(outputs_3[0].get_shape());
        const auto& rs_in0_shape = outputs_3[0].get_shape();
        const auto& rs_in1_shape = outputs_4[0].get_shape();
        int shift_dim = -1;
        size_t shift_offset = 1;
        for (size_t i = 0; i < rs_in1_shape.size(); ++i) {
            size_t dim = rs_in1_shape[i];
            if (dim != 1 && dim != rs_in0_shape[i]) {
                return false;
            }
            if (shift_dim != -1) {
                shift_offset *= rs_in0_shape[i];
            }
            if (dim == rs_in0_shape[i]) {
                shift_dim = static_cast<int>(i);
            }
        }
        if (shift_dim == -1)
            return false;
        for (size_t k = 0; k < rs_in0_shape_size; ++k) {
            size_t shift_idx = (k / shift_offset) % rs_in1_shape[shift_dim];
            int32_t shift_val = rs_in1[shift_idx];
            dst[k] = (rs_in0[k] >> shift_val);
        }

        std::shared_ptr<ov::Node> convert_1_node = nullptr;
        OutputVector outputs_7;
        if (pattern_map.find(convert_1) != pattern_map.end()) {
            convert_1_node = pattern_map.at(convert_1).get_node_shared_ptr();
            outputs_7.resize(convert_1_node->get_output_size());
            if (!convert_1_node->constant_fold(outputs_7, shifted_const->outputs())) {
                return false;
            }
        } else {
            auto convert_3_node = pattern_map.at(convert_3).get_node_shared_ptr();
            auto convert_4_node = pattern_map.at(convert_4).get_node_shared_ptr();
            auto convert_4_in_const =
                std::dynamic_pointer_cast<v0::Constant>(convert_4_node->get_input_node_shared_ptr(0));
            auto add_node = pattern_map.at(add).get_node_shared_ptr();
            OutputVector outputs_5(convert_3_node->get_output_size());
            if (!convert_3_node->constant_fold(outputs_5, shifted_const->outputs())) {
                return false;
            }
            OutputVector outputs_6(convert_4_node->get_output_size());
            if (!convert_4_node->constant_fold(outputs_6, convert_4_in_const->outputs())) {
                return false;
            }
            outputs_7.resize(add_node->get_output_size());
            OutputVector add_inputs(2);
            add_inputs[0] = outputs_5[0];
            add_inputs[1] = outputs_6[0];
            if (!add_node->constant_fold(outputs_7, add_inputs)) {
                return false;
            }
        }

        auto convert_2_node = pattern_map.at(convert_2).get_node_shared_ptr();
        auto convert_2_in_const = std::dynamic_pointer_cast<v0::Constant>(convert_2_node->get_input_node_shared_ptr(0));

        OutputVector outputs_8(convert_2_node->get_output_size());
        if (!convert_2_node->constant_fold(outputs_8, convert_2_in_const->outputs())) {
            return false;
        }

        OutputVector outputs_9(bitwise_and->get_output_size());

        const int8_t* and_in0 =
            std::dynamic_pointer_cast<v0::Constant>(outputs_7[0].get_node_shared_ptr())->get_data_ptr<int8_t>();
        const int8_t* and_in1 =
            std::dynamic_pointer_cast<v0::Constant>(outputs_8[0].get_node_shared_ptr())->get_data_ptr<int8_t>();
        auto masked_const = std::make_shared<v0::Constant>(element::i8, outputs_7[0].get_shape());
        auto masked_dst = const_cast<int8_t*>(reinterpret_cast<const int8_t*>(masked_const->get_data_ptr()));
        if (!masked_dst)
            return false;

        size_t and_in0_shape_size = shape_size(outputs_7[0].get_shape());
        // TODO: Bitwise and operation below might need to be
        // optimized to reduce FIL.
        int8_t mask = and_in1[0];
        for (size_t k = 0; k < and_in0_shape_size; ++k) {
            masked_dst[k] = (and_in0[k] & mask);
        }

        auto convert_to_u4 = std::make_shared<v0::Convert>(masked_const, element::u4);
        OutputVector outputs_10(convert_to_u4->get_output_size());
        if (!convert_to_u4->constant_fold(outputs_10, masked_const->outputs())) {
            return false;
        }

        auto new_convert =
            std::make_shared<v0::Convert>(outputs_10[0].get_node_shared_ptr(), bitwise_and->get_output_element_type(0));
        copy_runtime_info_and_name(bitwise_and, {new_convert}, {unsqueeze_1_node});
        replace_node(bitwise_and, new_convert);
        return true;
    };

    auto m = std::make_shared<Matcher>(bitwise_and, "ov::frontend::pytorch::pass::GPTQDecompressionReplacer");
    this->register_matcher(m, callback);
};

GPTQMultPatternReplacer::GPTQMultPatternReplacer() {
    const auto& const_1 = wrap_type<v0::Constant>();
    const auto& convert_1 = wrap_type<v0::Convert>({const_1});
    const auto& const_2 = wrap_type<v0::Constant>();
    const auto& convert_2 = wrap_type<v0::Convert>({const_2});
    const auto& add = wrap_type<v1::Add>({convert_1, convert_2});
    const auto& const_3 = wrap_type<v0::Constant>();
    const auto& add_or_convert = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{add, convert_1});
    const auto& reshape_1 = wrap_type<v1::Reshape>({add_or_convert, const_3});
    const auto& const_4 = wrap_type<v0::Constant>();
    const auto& convert_4 = wrap_type<v0::Convert>({const_4});
    const auto& const_5 = wrap_type<v0::Constant>();
    const auto& reshape_2 = wrap_type<v1::Reshape>({convert_4, const_5});
    const auto& subtract = wrap_type<v1::Subtract>({reshape_2, reshape_1});
    const auto& convert_3 = wrap_type<v0::Convert>({subtract});
    const auto& const_6 = wrap_type<v0::Constant>();
    const auto& const_7 = wrap_type<v0::Constant>();
    const auto& reshape_3 = wrap_type<v1::Reshape>({const_6, const_7});

    auto mult = wrap_type<v1::Multiply>({reshape_3, convert_3});

    ov::matcher_pass_callback callback = [=](Matcher& m) {
        auto mult = m.get_match_root();
        if (!mult) {
            return false;
        }
        const auto& pattern_map = m.get_pattern_value_map();
        std::shared_ptr<ov::Node> convert_2_node = nullptr;
        if (pattern_map.find(convert_2) != pattern_map.end()) {
            convert_2_node = pattern_map.at(convert_2).get_node_shared_ptr();
        }
        auto convert_1_node = pattern_map.at(convert_1).get_node_shared_ptr();
        auto convert_4_node = pattern_map.at(convert_4).get_node_shared_ptr();
        auto reshape_node = pattern_map.at(reshape_1).get_node_shared_ptr();
        auto reshape2_node = pattern_map.at(reshape_2).get_node_shared_ptr();
        auto reshape3_node = pattern_map.at(reshape_3).get_node_shared_ptr();
        // auto mult_node = pattern_map.at(mult).get_node_shared_ptr();

        auto add_input0_const = std::dynamic_pointer_cast<v0::Constant>(convert_1_node->get_input_node_shared_ptr(0));
        if (add_input0_const->get_element_type() != element::u4) {
            return false;
        }
        auto add_in0_ptr = add_input0_const->get_data_ptr<uint8_t>();
        uint32_t add_val = 0;
        if (convert_2_node) {
            auto convert_2_input_const =
                std::dynamic_pointer_cast<v0::Constant>(convert_2_node->get_input_node_shared_ptr(0));
            auto add_in1_ptr = convert_2_input_const->get_data_ptr<uint8_t>();
            if (!add_in1_ptr)
                return false;
            add_val = (uint32_t)(add_in1_ptr[0] & 0x0F);
        }
        const auto& add_in0_shape = add_input0_const->get_shape();
        const auto& static_shape_1 = reshape_node->get_shape();
        size_t add_in0_size = shape_size(add_in0_shape);
        auto add_replace_const = std::make_shared<v0::Constant>(element::f32, static_shape_1);
        auto add_replace_ptr = const_cast<float*>(reinterpret_cast<const float*>(add_replace_const->get_data_ptr()));

        if (!add_replace_ptr || (add_in0_size != shape_size(add_replace_const->get_shape()))) {
            return false;
        }

        for (size_t i = 0; i < add_in0_size; i++) {
            uint32_t val = read_u4_data(add_in0_ptr, i);
            val = (val + add_val) & 0x0F;
            add_replace_ptr[i] = (float)val;
        }

        const auto& static_shape_2 = reshape2_node->get_shape();
        auto reshape2_in0_const = std::dynamic_pointer_cast<v0::Constant>(convert_4_node->get_input_node_shared_ptr(0));
        auto sub_replace_const = std::make_shared<v0::Constant>(reshape2_in0_const->get_element_type(),
                                                                static_shape_2,
                                                                reshape2_in0_const->get_data_ptr<uint8_t>());
        auto new_convert_node = std::make_shared<v0::Convert>(sub_replace_const, element::f32);
        auto new_sub_node = std::make_shared<v1::Subtract>(new_convert_node, add_replace_const);

        const auto& static_shape_3 = reshape3_node->get_shape();
        auto reshape3_in0_const = std::dynamic_pointer_cast<v0::Constant>(reshape3_node->get_input_node_shared_ptr(0));
        auto mult_scale_const = std::make_shared<v0::Constant>(reshape3_in0_const->get_element_type(),
                                                               static_shape_3,
                                                               reshape3_in0_const->get_data_ptr<uint8_t>());
        auto new_mult_node = std::make_shared<v1::Multiply>(new_sub_node, mult_scale_const);
        replace_node(mult, new_mult_node);
        return true;
    };

    auto m = std::make_shared<Matcher>(mult, "ov::frontend::pytorch::pass::GPTQMultPatternReplacer");
    this->register_matcher(m, callback);
};

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
