// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/transformations/convert_transpose_interpolate_data_type.hpp"
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset4.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include "transformations/rt_info/disable_fp16_compression.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(pass::ConvertTransposePrecision, "ConvertTransposePrecision", 0);

pass::ConvertTransposePrecision::ConvertTransposePrecision() {
    auto transpose_pattern = pattern::wrap_type<opset8::Transpose>();

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto transpose_node = std::dynamic_pointer_cast<opset8::Transpose>(m.get_match_root());
        if (!transpose_node)
            return false;

        auto dst_type = element::Type_t::f16;
        auto src_type = element::Type_t::u8;
        ov::Output<ov::Node> new_in  = transpose_node->input_value(0);
        ov::Output<ov::Node> new_out = transpose_node->outputs()[0];

        if (new_in.get_element_type() != src_type && new_out.get_element_type() != src_type)
            return false;

        auto convert = std::make_shared<ngraph::opset1::Convert>(new_in, dst_type);
        auto transpose_node_fp16 = make_shared<opset8::Transpose>(convert,
                                                                  transpose_node->input_value(1));
        ngraph::copy_runtime_info(transpose_node, transpose_node_fp16);
        ngraph::replace_node(transpose_node, transpose_node_fp16);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(transpose_pattern, "ConvertTransposePrecision");
    register_matcher(m, callback);
}
