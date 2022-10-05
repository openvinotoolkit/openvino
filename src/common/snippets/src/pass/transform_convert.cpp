// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/remarks.hpp"
#include <snippets/itt.hpp>

#include "snippets/pass/transform_convert.hpp"
#include "snippets/snippets_isa.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

ngraph::snippets::pass::TransformConvertToConvertTruncation::TransformConvertToConvertTruncation() {
    MATCHER_SCOPE(TransformConvertToConvertTruncation);
    auto convert = std::make_shared<pattern::op::Label>(pattern::any_input(),
        [](const std::shared_ptr<const Node> &n) {
            return ov::is_type<ngraph::opset1::Convert>(n) &&
                !ov::is_type<op::ConvertTruncation>(n) &&
                !ov::is_type<op::ConvertSaturation>(n);
        });

    register_matcher(std::make_shared<ngraph::pattern::Matcher>(
        ngraph::pattern::wrap_type<ngraph::opset1::Convert>(), matcher_name),
            [this](ngraph::pattern::Matcher &m) {
            OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::op::TransformConvertToConvertTruncation")
            const auto root = m.get_match_root();
            const auto convert = ngraph::as_type_ptr<ngraph::opset1::Convert>(root);
            auto convert_truncation = std::make_shared<op::ConvertTruncation>(convert->get_input_source_output(0),
                                                                              convert->get_destination_type());
            convert_truncation->set_friendly_name(convert->get_friendly_name());
            ngraph::copy_runtime_info(convert, convert_truncation);
            ngraph::replace_node(convert, convert_truncation);

            return true;
        });
}
