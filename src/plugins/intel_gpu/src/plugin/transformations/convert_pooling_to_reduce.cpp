// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_pooling_to_reduce.hpp"

#include <algorithm>
#include <memory>
#include <vector>

#include <ngraph/opsets/opset9.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>

ov::intel_gpu::ConvertAvgPoolingToReduce::ConvertAvgPoolingToReduce() {
    // Check all AvgPool nodes
    auto m = std::make_shared<ngraph::pattern::Matcher>(ngraph::pattern::wrap_type<ngraph::opset9::AvgPool>(), "ConvertAvgPoolingToReduce");
    register_matcher(m, [&](ngraph::pattern::Matcher& m) {
        auto pool = std::dynamic_pointer_cast<ngraph::opset9::AvgPool>(m.get_match_root());
        if (!pool || transformation_callback(pool)) {
            return false;
        }

        auto kernel = pool->get_kernel();
        auto pads_begin = pool->get_pads_begin();
        auto pads_end = pool->get_pads_end();

        int64_t rank = pool->get_input_partial_shape(0).size();
        auto input_shape = pool->get_input_shape(0);
        // Check if input spatial size is same with kernel size.
        bool has_same_spatial_size = rank > 2 && std::equal(input_shape.begin() + 2, input_shape.end(), kernel.begin());
        // Check if pads are zeros.
        bool no_padding =
            std::count(pads_begin.begin(), pads_begin.end(), 0) == static_cast<int64_t>(pads_begin.size()) &&
            std::count(pads_end.begin(), pads_end.end(), 0) == static_cast<int64_t>(pads_end.size());

        if (!has_same_spatial_size || !no_padding) {
            return false;
        }

        std::vector<int64_t> axes_shape(rank - 2);
        std::iota(axes_shape.begin(), axes_shape.end(), 2);

        auto reduce = std::make_shared<ngraph::opset9::ReduceMean>(
            pool->input_value(0),
            ngraph::opset9::Constant::create(ngraph::element::i64, ngraph::Shape{axes_shape.size()}, axes_shape),
            true);

        reduce->set_friendly_name(pool->get_friendly_name() + "/Reduce");
        copy_runtime_info(pool, reduce);
        replace_node(pool, reduce);

        return true;
    });
}
