// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/convert_constants.hpp"

#include <memory>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type.hpp"
#include "openvino/op/constant.hpp"
#include "snippets/itt.hpp"
#include "snippets/op/scalar.hpp"

bool ov::snippets::pass::ConvertConstantsToScalars::run_on_model(const std::shared_ptr<ov::Model>& m) {
    RUN_ON_MODEL_SCOPE(ConvertConstantsToScalars);
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::op::ConvertConstantsToScalars")

    bool changed = false;
    for (const auto& node : m->get_ordered_ops()) {
        auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node);
        if (!constant) {
            continue;
        }
        if (ov::shape_size(constant->get_output_shape(0)) != 1) {
            continue;
        }
        //  Note: all Constants {1,1,1,1} are converted to Scalar {1}
        //  This simplifies shape inference and avoids rank increases by [1,1,1,1] constants.
        auto scalar = std::make_shared<snippets::op::Scalar>(ov::op::v0::Constant(*constant, ov::Shape{1}));
        scalar->set_friendly_name(constant->get_friendly_name());
        ov::copy_runtime_info(constant, scalar);
        ov::replace_node(constant, scalar);
        changed = true;
    }
    return changed;
}
