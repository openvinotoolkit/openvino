// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "decompose_integer_divide.hpp"

#include <memory>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/type.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/floor.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov::intel_cpu {

DecomposeIntegerDivide::DecomposeIntegerDivide() {
    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(ov::pass::pattern::wrap_type<ov::op::v1::Divide>(),
                                                                  "DecomposeIntegerDivide"),
                     [](ov::pass::pattern::Matcher& m) {
                         auto divide = ov::as_type_ptr<ov::op::v1::Divide>(m.get_match_root());
                         if (!divide) {
                             return false;
                         }
                         if (!divide->get_element_type().is_integral_number()) {
                             return false;
                         }

                         auto new_divide =
                             std::make_shared<ov::op::v1::Divide>(divide->input_value(0), divide->input_value(1));
                         auto new_floor = std::make_shared<ov::op::v0::Floor>(new_divide);
                         new_floor->set_friendly_name(divide->get_friendly_name());
                         ov::copy_runtime_info(divide, new_floor);
                         ov::replace_node(divide, new_floor);
                         return true;
                     });
}

}  // namespace ov::intel_cpu
