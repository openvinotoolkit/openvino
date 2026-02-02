// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/qdq_stripping.hpp"

#include <memory>

#include "itt.hpp"
#include "low_precision/common/ie_lpt_exception.hpp"
#include "low_precision/lpt_itt.hpp"
#include "low_precision/network_helper.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/log.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
namespace pass {
namespace low_precision {

FQStrippingTransformation::FQStrippingTransformation(const std::set<size_t>& levels_to_strip, bool replace_with_clamp) {
    MATCHER_SCOPE(FQStrippingTransformation);
    auto is_scalar = [](const Output<Node>& output) -> bool {
        return ov::shape_size(output.get_shape()) == 1;
    };
    auto input_low_m = pattern::wrap_type<ov::op::v0::Constant>(is_scalar);
    auto input_high_m = pattern::wrap_type<ov::op::v0::Constant>(is_scalar);
    auto output_low_m = pattern::wrap_type<ov::op::v0::Constant>(is_scalar);
    auto output_high_m = pattern::wrap_type<ov::op::v0::Constant>(is_scalar);
    auto fq_m = pattern::wrap_type<ov::op::v0::FakeQuantize>(
        {pattern::any_input(), input_low_m, input_high_m, output_low_m, output_high_m});

    ov::graph_rewrite_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto node = ov::as_type_ptr<ov::op::v0::FakeQuantize>(pattern_map.at(fq_m).get_node_shared_ptr());
        if (!node) {
            return false;
        }

        const size_t levels = node->get_levels();
        if (!levels_to_strip.count(levels)) {
            return false;
        }

        auto input = node->get_input_node_shared_ptr(0);
        auto input_low = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(input_low_m).get_node_shared_ptr());
        auto input_high = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(input_high_m).get_node_shared_ptr());
        auto output_low = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(output_low_m).get_node_shared_ptr());
        auto output_high = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(output_high_m).get_node_shared_ptr());

        if (!input_low || !input_high || !output_low || !output_high) {
            return false;
        }
        auto constants_are_equal = [](const std::shared_ptr<ov::op::v0::Constant>& lhs,
                                      const std::shared_ptr<ov::op::v0::Constant>& rhs) -> bool {
            auto equal =
                ov::as_type_ptr<ov::op::v0::Constant>(ov::op::util::make_try_fold<ov::op::v1::Equal>(lhs, rhs));
            OPENVINO_ASSERT(equal && ov::shape_size(equal->get_shape()) == 1,
                            "constants_are_equal expects scalar constant as a comparison result");
            return equal->get_vector<bool>()[0];
        };

        if (!constants_are_equal(input_low, output_low) || !constants_are_equal(input_high, output_high)) {
            return false;
        }

        bool res = false;
        if (replace_with_clamp) {
            auto clamp = std::make_shared<ov::op::v0::Clamp>(input->output(0),
                                                             output_low->cast_vector<double>()[0],
                                                             output_high->cast_vector<double>()[0]);
            res = replace_node_update_name(node, clamp);
        } else {
            res = replace_output_update_name(node->output(0), node->input_value(0));
        }
        OPENVINO_ASSERT(res, "FQ stripping failed");
        return res;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(fq_m, matcher_name);
    this->register_matcher(m, callback);
}

}  // namespace low_precision
}  // namespace pass
}  // namespace ov