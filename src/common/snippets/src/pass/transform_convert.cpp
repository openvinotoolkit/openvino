// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/itt.hpp"

#include "snippets/pass/transform_convert.hpp"
#include "snippets/snippets_isa.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

ov::snippets::pass::TransformConvertToConvertTruncation::TransformConvertToConvertTruncation() {
    MATCHER_SCOPE(TransformConvertToConvertTruncation);
    auto convert = std::make_shared<ov::pass::pattern::op::Label>(
        ov::pass::pattern::any_input(),
        [](const std::shared_ptr<const Node>& n) {
            return ov::is_type<ov::opset1::Convert>(n) &&
                   !ov::is_type_any_of<op::ConvertTruncation, op::ConvertSaturation>(n);
        });

    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(
        ov::pass::pattern::wrap_type<ov::opset1::Convert>(), matcher_name), [](ov::pass::pattern::Matcher &m) {
            OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::op::TransformConvertToConvertTruncation")
            const auto root = m.get_match_root();
            const auto convert = ov::as_type_ptr<ov::opset1::Convert>(root);
            auto convert_truncation = std::make_shared<op::ConvertTruncation>(convert->get_input_source_output(0),
                                                                              convert->get_destination_type());
            convert_truncation->set_friendly_name(convert->get_friendly_name());
            ov::copy_runtime_info(convert, convert_truncation);
            ov::replace_node(convert, convert_truncation);

            return true;
        });
}
