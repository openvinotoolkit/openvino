// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_pooling_to_reduce.hpp"

#include <algorithm>
#include <memory>
#include <vector>

#include "openvino/op/avg_pool.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/core/rt_info.hpp"

ov::intel_gpu::ConvertAvgPoolingToReduce::ConvertAvgPoolingToReduce() {
    // Check all AvgPool nodes
    auto m = std::make_shared<ov::pass::pattern::Matcher>(ov::pass::pattern::wrap_type<ov::op::v1::AvgPool>(), "ConvertAvgPoolingToReduce");
    register_matcher(m, [&](ov::pass::pattern::Matcher& m) {
        auto pool = ov::as_type_ptr<ov::op::v1::AvgPool>(m.get_match_root());
        if (!pool || transformation_callback(pool)) {
            return false;
        }

        auto kernel = pool->get_kernel();
        auto pads_begin = pool->get_pads_begin();
        auto pads_end = pool->get_pads_end();

        auto input = pool->input_value(0);
        const auto input_shape = input.get_partial_shape();
        if (input_shape.is_dynamic() || input_shape.rank().is_dynamic()) {
            return false;
        }
        const auto rank = input_shape.rank().get_length();
        // Check if input spatial size is same with kernel size.
        bool has_same_spatial_size = rank > 2 && std::equal(input_shape.end() - (rank - 2), input_shape.end(), kernel.end() - (rank - 2));
        // Check if pads are zeros.
        bool no_padding =
            std::count(pads_begin.begin(), pads_begin.end(), 0) == static_cast<int64_t>(pads_begin.size()) &&
            std::count(pads_end.begin(), pads_end.end(), 0) == static_cast<int64_t>(pads_end.size());

        if (!has_same_spatial_size || !no_padding) {
            return false;
        }

        std::vector<int64_t> axes_shape(rank - 2);
        std::iota(axes_shape.begin(), axes_shape.end(), 2);

        auto reduce = std::make_shared<ov::op::v1::ReduceMean>(
            pool->input_value(0),
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{axes_shape.size()}, axes_shape),
            true);

        reduce->set_friendly_name(pool->get_friendly_name());
        copy_runtime_info(pool, reduce);
        replace_node(pool, reduce);

        return true;
    });
}
