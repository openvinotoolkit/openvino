// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/transform_convert.hpp"

#include <memory>

#include "openvino/core/except.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/type.hpp"
#include "openvino/opsets/opset1.hpp"
#include "snippets/itt.hpp"
#include "snippets/op/convert_saturation.hpp"
#include "snippets/op/convert_truncation.hpp"

bool ov::snippets::pass::TransformConvertToConvertTruncation::run_on_model(const std::shared_ptr<ov::Model>& m) {
    RUN_ON_MODEL_SCOPE(TransformConvertToConvertTruncation);
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::op::TransformConvertToConvertTruncation")

    bool changed = false;
    // Take a snapshot to be safe while replacing nodes
    const auto ops = m->get_ordered_ops();
    for (const auto& node : ops) {
        // Only handle original Convert ops; skip our custom ones
        if (!ov::is_type<ov::opset1::Convert>(node) ||
            ov::is_type_any_of<ov::snippets::op::ConvertTruncation, ov::snippets::op::ConvertSaturation>(node)) {
            continue;
        }
        auto convert = ov::as_type_ptr<ov::opset1::Convert>(node);
        OPENVINO_ASSERT(convert, "Convert op is invalid");
        auto convert_truncation =
            std::make_shared<ov::snippets::op::ConvertTruncation>(convert->get_input_source_output(0),
                                                                  convert->get_destination_type());
        convert_truncation->set_friendly_name(convert->get_friendly_name());
        ov::copy_runtime_info(convert, convert_truncation);
        ov::replace_node(convert, convert_truncation);
        changed = true;
    }
    return changed;
}
