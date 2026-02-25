// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/dropout_with_random_uniform_replacer.hpp"

#include <memory>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/floor.hpp"
#include "openvino/op/random_uniform.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace v0 = ov::op::v0;
namespace v8 = ov::op::v8;
namespace op_util = ov::op::util;

namespace ov::pass {

DropoutWithRandomUniformReplacer::DropoutWithRandomUniformReplacer() {
    MATCHER_SCOPE(DropoutWithRandomUniformReplacer);
    const auto shape_pattern = pattern::any_input();
    const auto ru_min_const_pattern = pattern::wrap_type<v0::Constant>();
    const auto ru_max_const_pattern = pattern::wrap_type<v0::Constant>();
    const auto random_uniform_pattern =
        pattern::wrap_type<v8::RandomUniform>({shape_pattern, ru_min_const_pattern, ru_max_const_pattern},
                                              pattern::consumers_count(1));

    const auto optional_convert = pattern::optional<v0::Convert>(random_uniform_pattern);
    const auto add_const_pattern = pattern::wrap_type<v0::Constant>();

    const auto add_pattern = pattern::wrap_type<ov::op::v1::Add>({optional_convert, add_const_pattern});

    const auto floor_pattern = pattern::wrap_type<v0::Floor>({add_pattern});

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto random_uniform = pattern_map.at(random_uniform_pattern);
        const auto shape_of = pattern_map.at(shape_pattern);
        const auto ru = ov::as_type_ptr<v8::RandomUniform>(random_uniform.get_node_shared_ptr());
        if (!ru)
            return false;
        if (!ru->get_out_type().is_real())
            return false;

        auto min_const_value =
            ov::as_type_ptr<v0::Constant>(pattern_map.at(ru_min_const_pattern).get_node_shared_ptr());
        auto max_const_value =
            ov::as_type_ptr<v0::Constant>(pattern_map.at(ru_max_const_pattern).get_node_shared_ptr());
        auto add_const_value = ov::as_type_ptr<v0::Constant>(pattern_map.at(add_const_pattern).get_node_shared_ptr());

        bool valid_constant_values = op_util::has_constant_value<double>(min_const_value, 0.0) &&
                                     op_util::has_constant_value<double>(max_const_value, 1.0);
        if (!valid_constant_values)
            return false;

        if (!add_const_value)
            return false;

        auto add_const_vector = add_const_value->cast_vector<double>();
        if (add_const_vector.size() > 1)
            return false;

        // Add const should have zero fractional part
        if (add_const_vector[0] - std::round(add_const_vector[0]) != 0.0)
            return false;

        const auto broadcast_const = v0::Constant::create(ru->get_out_type(), Shape{}, {0.5});
        const auto broadcast = std::make_shared<ov::op::v3::Broadcast>(broadcast_const, shape_of);

        broadcast->set_friendly_name(ru->get_friendly_name());
        copy_runtime_info(ru, broadcast);
        ov::replace_node(ru, broadcast);

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(floor_pattern, matcher_name);
    this->register_matcher(m, callback);
}

}  // namespace ov::pass
