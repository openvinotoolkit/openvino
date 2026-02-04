// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_gelu.hpp"

#include <memory>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/erf.hpp"
#include "openvino/op/gelu.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

using ov::pass::pattern::Matcher;

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
ov::pass::ConvertGELU::ConvertGELU() {
    MATCHER_SCOPE(ConvertGELU);
    auto gelu = ov::pass::pattern::wrap_type<v0::Gelu>();

    matcher_pass_callback callback = [this](Matcher& m) {
        auto gelu = ov::as_type_ptr<v0::Gelu>(m.get_match_root());
        if (!gelu || transformation_callback(gelu))
            return false;
        auto input = gelu->input_value(0);
        auto input_type = input.get_element_type();

        // f(x) = 0.5 * x * (1.0 + erf( x / sqrt(2.0) )
        auto mul = std::make_shared<v1::Multiply>(input, v0::Constant::create(input_type, Shape{}, {0.5}));
        auto sq2 = std::make_shared<v0::Sqrt>(v0::Constant::create(input_type, Shape{}, {2.0}));
        auto div = register_new_node<v1::Divide>(input, sq2);  // can be decomposed
        auto erf = std::make_shared<v0::Erf>(div);
        auto add = std::make_shared<v1::Add>(erf, v0::Constant::create(input_type, Shape{}, {1.0}));
        auto res = std::make_shared<v1::Multiply>(mul, add);

        res->set_friendly_name(gelu->get_friendly_name());
        ov::copy_runtime_info(gelu, {mul, sq2, div, erf, add, res});
        ov::replace_node(gelu, res);
        return true;
    };

    auto m = std::make_shared<Matcher>(gelu, matcher_name);
    register_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
}
