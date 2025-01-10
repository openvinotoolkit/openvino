// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/reduce_reshape_fusion.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/util/arithmetic_reductions_keep_dims.hpp"
#include "openvino/op/util/binary_elementwise_logical.hpp"
#include "openvino/op/util/logical_reduction_keep_dims.hpp"
#include "openvino/op/util/reduction_base.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

ov::pass::ReduceReshapeFusion::ReduceReshapeFusion() {
    MATCHER_SCOPE(ReduceReshapeFusion);

    const auto reduce_axes = pattern::wrap_type<ov::op::v0::Constant>();
    const auto reduce = pattern::wrap_type<op::util::ArithmeticReductionKeepDims, op::util::LogicalReductionKeepDims>(
        {pattern::any_input(), reduce_axes},
        pattern::consumers_count(1));
    const auto reshape =
        pattern::wrap_type<ov::op::v1::Reshape>({reduce, pattern::any_input()}, pattern::has_static_shape());

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto& pattern_map = m.get_pattern_value_map();
        auto reshape_node = pattern_map.at(reshape).get_node_shared_ptr();
        const auto reduce_node = ov::as_type_ptr<op::util::ReductionBase>(pattern_map.at(reduce).get_node_shared_ptr());
        if (!reduce_node) {
            return false;
        }
        if (reduce_node->get_output_partial_shape(0).is_dynamic()) {
            return false;
        }
        const bool keep_dims = reduce_node->get_keep_dims();

        if (keep_dims) {
            return false;
        }

        const auto reduce_axes_val = reduce_node->get_reduction_axes().to_vector();
        const auto& reshape_shape = reshape_node->get_shape();

        auto reduce_shape = reduce_node->get_shape();
        for (const auto& axis : reduce_axes_val) {
            reduce_shape.insert(std::next(std::begin(reduce_shape), axis), 1);
        }

        if (reduce_shape != reshape_shape) {
            return false;
        }

        if (auto arithmetic_reduce_node = ov::as_type_ptr<op::util::ArithmeticReductionKeepDims>(reduce_node)) {
            arithmetic_reduce_node->set_keep_dims(true);
        } else if (auto logical_reduce_node = ov::as_type_ptr<op::util::LogicalReductionKeepDims>(reduce_node)) {
            logical_reduce_node->set_keep_dims(true);
        }
        reduce_node->validate_and_infer_types();
        reduce_node->set_friendly_name(reshape_node->get_friendly_name());
        copy_runtime_info(reshape_node, reduce_node);
        replace_node(m.get_match_root(), reduce_node);

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(reshape, matcher_name);
    register_matcher(m, callback);
}
