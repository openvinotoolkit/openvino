// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/softsign_decomposition.hpp"

#include <memory>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/softsign.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

ov::pass::SoftSignDecomposition::SoftSignDecomposition() {
    MATCHER_SCOPE(SoftSignDecomposition);
    auto softsign = pattern::wrap_type<ov::op::v9::SoftSign>();
    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        auto m_softsign = m.get_match_root();

        if (transformation_callback(m_softsign)) {
            return false;
        }

        const auto input = m_softsign->input_value(0);
        const auto& data_type = m_softsign->get_input_element_type(0);
        auto abs = std::make_shared<ov::op::v0::Abs>(input);
        auto constant = ov::op::v0::Constant::create(data_type, ov::Shape{1}, {1});
        auto add = std::make_shared<ov::op::v1::Add>(abs, constant);
        auto div = std::make_shared<ov::op::v1::Divide>(input, add);

        replace_node(m_softsign, div);
        copy_runtime_info(m_softsign, {abs, add, div});
        div->set_friendly_name(m_softsign->get_friendly_name());

        return true;
    };
    auto m = std::make_shared<ov::pass::pattern::Matcher>(softsign, matcher_name);
    register_matcher(m, callback);
}
