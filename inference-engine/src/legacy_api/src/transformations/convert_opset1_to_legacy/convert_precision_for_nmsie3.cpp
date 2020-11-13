// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "legacy/transformations/convert_opset1_to_legacy/convert_precision_for_nmsie3.hpp"

#include <memory>
#include <vector>

#include <legacy/ngraph_ops/nms_ie.hpp>
#include <ngraph_ops/type_relaxed.hpp>

using namespace ngraph;

NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvertNMSIE3Precision, "ConvertNMSIE3Precision", 0);

bool ngraph::pass::ConvertNMSIE3Precision::run_on_function(std::shared_ptr<ngraph::Function> f) {
    for (auto &node : f->get_ordered_ops()) {
        auto nms_ie = std::dynamic_pointer_cast<ngraph::op::NonMaxSuppressionIE3>(node);
        if (nms_ie && nms_ie->m_output_type == element::i64) {
            nms_ie->m_output_type = element::i32;
        }
    }
    return true;
}
