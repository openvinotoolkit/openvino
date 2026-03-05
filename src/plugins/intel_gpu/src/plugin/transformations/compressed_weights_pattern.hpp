#pragma once

using namespace ov::pass::pattern;
#define FC_COMPRESSED_WEIGHT_PATTERN\
        auto compressed_constant = [](const ov::Output<ov::Node>& output) {\
            return (output.get_element_type() == ov::element::u8 || output.get_element_type() == ov::element::i8 ||\
                    output.get_element_type() == ov::element::u4 || output.get_element_type() == ov::element::i4);\
        };\
        \
        auto reshape_squeeze = [](const ov::Output<ov::Node>& output) {\
            auto in_ps = output.get_node()->get_input_partial_shape(0);\
            auto out_ps = output.get_node()->get_output_partial_shape(0);\
            return in_ps.rank().is_static() && out_ps.rank().is_static() &&\
                   ((in_ps.size() == 3 && out_ps.size() == 2) || (in_ps.size() == 4 && out_ps.size() == 3));\
        };\
        \
        auto compressed_weights_m = wrap_type<ov::op::v0::Constant>(compressed_constant);\
        auto convert_m = wrap_type<ov::op::v0::Convert>({compressed_weights_m});\
\
        auto sub_const_m = wrap_type<ov::op::v0::Constant>();\
        auto sub_convert_const_m = wrap_type<ov::op::v0::Convert>({sub_const_m});\
        auto sub_with_convert_m = wrap_type<ov::op::v1::Subtract>({convert_m, sub_convert_const_m});\
        auto sub_no_convert_m = wrap_type<ov::op::v1::Subtract>({convert_m, sub_const_m});\
        auto subtract_m = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{sub_with_convert_m, sub_no_convert_m});\
\
        auto mul_const_m = wrap_type<ov::op::v0::Constant>();\
        auto mul_with_sub_m = wrap_type<ov::op::v1::Multiply>({subtract_m, mul_const_m});\
        auto mul_no_sub_m = wrap_type<ov::op::v1::Multiply>({convert_m, mul_const_m});\
        auto mul_m = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{mul_with_sub_m, mul_no_sub_m});\
\
        auto reshape_const_m = wrap_type<ov::op::v0::Constant>();\
        auto reshape_m = wrap_type<ov::op::v1::Reshape>({mul_m, reshape_const_m}, reshape_squeeze);\
        auto convert_reshape_m = wrap_type<ov::op::v0::Convert>({reshape_m});\
\
        auto mul2_const_m = wrap_type<ov::op::v0::Constant>();\
        auto mul2_m = wrap_type<ov::op::v1::Multiply>({reshape_m, mul2_const_m});\
\
        auto transpose_input = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{reshape_m, mul_m});\
        auto transpose_const_m = wrap_type<ov::op::v0::Constant>();\
        auto transpose_m = wrap_type<ov::op::v1::Transpose>({transpose_input, transpose_const_m});\
\
        auto compressed_weights_input_m =\
            std::make_shared<ov::pass::pattern::op::Or>(ov::OutputVector{reshape_m, convert_reshape_m, transpose_m, mul_m, mul2_m});
