// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/transform_convert.hpp"

#include <memory>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/type.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "snippets/itt.hpp"
#include "snippets/op/convert_truncation.hpp"
#include "snippets/op/subgraph.hpp"

ov::snippets::pass::TransformConvertToConvertTruncation::TransformConvertToConvertTruncation() {
    MATCHER_SCOPE(TransformConvertToConvertTruncation);
    auto matcher =
        std::make_shared<ov::pass::pattern::Matcher>(ov::pass::pattern::wrap_type<ov::snippets::op::Subgraph>(),
                                                     matcher_name);

    register_matcher(matcher, [this](ov::pass::pattern::Matcher& m) {
        OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform,
                           "Snippets::op::TransformConvertToConvertTruncation")
        const auto subgraph = ov::as_type_ptr<ov::snippets::op::Subgraph>(m.get_match_root());
        if (!subgraph || this->transformation_callback(subgraph)) {
            return false;
        }

        auto body = subgraph->body_ptr();
        bool modified = false;
        for (const auto& node : body->get_ordered_ops()) {
            const auto convert = ov::as_type_ptr<ov::opset1::Convert>(node);
            if (!convert) {
                continue;
            }
            auto convert_truncation =
                std::make_shared<ov::snippets::op::ConvertTruncation>(convert->get_input_source_output(0),
                                                                      convert->get_destination_type());
            convert_truncation->set_friendly_name(convert->get_friendly_name());
            ov::copy_runtime_info(convert, convert_truncation);
            ov::replace_node(convert, convert_truncation);
            modified = true;
        }
        return modified;
    });
}
