// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "itt.hpp"
#include "transformations/op_conversions/convert_reduce_to_pooling.hpp"

NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvertReduceToPooling, "ConvertReduceToPooling", 0);
NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvertReduceMeanToPooling, "ConvertReduceMeanToPooling", 0);
NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvertReduceMaxToPooling, "ConvertReduceMaxToPooling", 0);
NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvertReduceSumToPooling, "ConvertReduceSumToPooling", 0);

ngraph::pass::ConvertReduceMeanToPooling::ConvertReduceMeanToPooling() {
    MATCHER_SCOPE(ConvertReduceMeanToPooling);
    auto m = std::make_shared<ngraph::pattern::Matcher>(ngraph::pattern::wrap_type<opset1::ReduceMean>({pattern::any_input(pattern::has_static_shape()),
        pattern::wrap_type<opset1::Constant>()},
        pattern::has_static_shape()), matcher_name);
    register_matcher(m, convert_reduce_to_pooling<opset1::ReduceMean>());
}
ngraph::pass::ConvertReduceMaxToPooling::ConvertReduceMaxToPooling() {
    MATCHER_SCOPE(ConvertReduceMaxToPooling);
    auto m = std::make_shared<ngraph::pattern::Matcher>(
                                                        ngraph::pattern::wrap_type<opset1::ReduceMax>({pattern::any_input(pattern::has_static_shape()),
                                                            pattern::wrap_type<opset1::Constant>()},
                                                            pattern::has_static_shape()), matcher_name);
    register_matcher(m, convert_reduce_to_pooling<opset1::ReduceMax>());
}
ngraph::pass::ConvertReduceSumToPooling::ConvertReduceSumToPooling() {
    MATCHER_SCOPE(ConvertReduceSumToPooling);
    auto m = std::make_shared<ngraph::pattern::Matcher>(
                                                        ngraph::pattern::wrap_type<opset1::ReduceSum>({pattern::any_input(pattern::has_static_shape()),
                                                            pattern::wrap_type<opset1::Constant>()},
                                                            pattern::has_static_shape()), matcher_name);
    register_matcher(m, convert_reduce_to_pooling<opset1::ReduceSum>());
}
