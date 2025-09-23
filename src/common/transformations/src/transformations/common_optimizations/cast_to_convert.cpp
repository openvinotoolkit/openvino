// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/cast_to_convert.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov::op;
using namespace ov::pass::pattern;

ov::pass::CastToConvert::CastToConvert() {
    MATCHER_SCOPE(CastToConvert);

    auto reshape = wrap_type<v1::Reshape>({any_input(), any_input()});
    auto reducemean = wrap_type<v1::ReduceMean>({reshape, any_input()});
    auto convert = wrap_type<v16::Convert>({reducemean});
    auto result = wrap_type<v0::Result>({convert});

    ov::matcher_pass_callback callback = [=](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto convert_node = pattern_map.at(convert).get_node_shared_ptr();
        auto convert_op = ov::as_type_ptr<v16::Convert>(convert_node);
        if (!convert_op)
            return false;

        if (!(convert_op->get_no_clamp() && convert_op->get_use_rounding()))
            return false;

        auto input = convert_op->input_value(0);
        auto dest_type = convert_op->get_destination_type();
        auto new_convert = std::make_shared<v0::Convert>(input, dest_type);
        new_convert->set_friendly_name(convert_op->get_friendly_name());
        copy_runtime_info(convert_op, new_convert);
        replace_node(convert_op, new_convert);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(result, matcher_name);
    this->register_matcher(m, callback);
}
