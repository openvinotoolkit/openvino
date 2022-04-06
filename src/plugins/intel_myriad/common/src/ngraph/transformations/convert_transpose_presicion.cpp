// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/transformations/convert_transpose_presicion.hpp"
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset4.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

using namespace std;

namespace vpu {

ConvertTransposePrecision::ConvertTransposePrecision() {
    auto transpose_pattern = ngraph::pattern::wrap_type<ngraph::opset8::Transpose>();

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto transpose_node = std::dynamic_pointer_cast<ngraph::opset8::Transpose>(m.get_match_root());
        if (!transpose_node)
            return false;

        auto dst_type = ngraph::element::Type_t::f16;
        auto src_type = ngraph::element::Type_t::u8;
        ov::Output<ov::Node> new_in  = transpose_node->input_value(0);
        ov::Output<ov::Node> new_out = transpose_node->outputs()[0];

        if (new_in.get_element_type() != src_type && new_out.get_element_type() != src_type)
            return false;

        auto convert = std::make_shared<ngraph::opset1::Convert>(new_in, dst_type);
        auto transpose_node_fp16 = make_shared<ngraph::opset8::Transpose>(convert,
                                                                          transpose_node->input_value(1));
        transpose_node_fp16->set_friendly_name(transpose_node->get_friendly_name());
        ngraph::copy_runtime_info(transpose_node, transpose_node_fp16);
        ngraph::replace_node(transpose_node, transpose_node_fp16);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(transpose_pattern, "ConvertTransposePrecision");
    register_matcher(m, callback);
}
} // namespace vpu
