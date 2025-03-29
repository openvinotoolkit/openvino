// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_convertlike.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

using namespace ov;

ov::pass::ConvertConvertLike::ConvertConvertLike() {
    MATCHER_SCOPE(ConvertConvertLike);

    auto convertlike = pattern::wrap_type<ov::op::v1::ConvertLike>();

    matcher_pass_callback callback = [](pattern::Matcher& m) {
        auto cvtlike = ov::as_type_ptr<ov::op::v1::ConvertLike>(m.get_match_root());
        if (!cvtlike) {
            return false;
        }

        auto like = cvtlike->input_value(1);
        const element::Type& dest_type = like.get_element_type();
        if (dest_type == element::dynamic)
            return false;

        auto cvt = std::make_shared<ov::op::v0::Convert>(cvtlike->input_value(0), dest_type);

        cvt->set_friendly_name(cvtlike->get_friendly_name());
        copy_runtime_info(cvtlike, cvt);
        replace_node(cvtlike, cvt);
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(convertlike, matcher_name);
    this->register_matcher(m, callback);
}
