// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_program.h"
#include "cldnn_common_utils.h"
#include "transformations/utils/utils.hpp"

#include "ngraph/op/pad.hpp"

#include "api/border.hpp"

namespace CLDNNPlugin {

static cldnn::border_type GetBorderType(ngraph::op::PadMode mode) {
    switch (mode) {
        case ngraph::op::PadMode::CONSTANT: return cldnn::border_type::constant;
        case ngraph::op::PadMode::EDGE: return cldnn::border_type::edge;
        case ngraph::op::PadMode::REFLECT: return cldnn::border_type::mirror_101;
        case ngraph::op::PadMode::SYMMETRIC: return cldnn::border_type::mirror;
        default: IE_THROW() << "Invalid border mode " << mode << " in layer ";
    }
    return cldnn::border_type::constant;
}

static std::vector<int32_t> GetPermuteOrder(const ngraph::CoordinateDiff& ie_order) {
    std::vector<int32_t> cldnn_order(ie_order.begin(), ie_order.end());

    // 1. Align to min. 4 sizes
    if (cldnn_order.size() < 4) {
        const auto zeros_to_add = 4 - ie_order.size();
        cldnn_order.insert(cldnn_order.end(), zeros_to_add, 0);
    }

    // 2. Swap spatial positions
    for (int i = 0; i < (cldnn_order.size() - 2) / 2; i++) {
        std::swap(cldnn_order[2 + i], cldnn_order[1 + cldnn_order.size() - (2 + i)]);
    }

    return cldnn_order;
}

void CreatePadOp(Program& p, const std::shared_ptr<ngraph::op::v1::Pad>& op) {
    p.ValidateInputs(op, {3, 4});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    auto pads_begin = cldnn::tensor(GetPermuteOrder(op->get_pads_begin()), 0);
    auto pads_end = cldnn::tensor(GetPermuteOrder(op->get_pads_end()), 0);
    float pad_value = 0.f;

    if (op->get_input_size() == 4) {
        auto const_node = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(op->get_input_node_shared_ptr(3));
        if (!const_node) {
            IE_THROW() << "Unsupported const node type in " << op->get_friendly_name() << " (" << op->get_type_name() << ")";
        }
        if (!ngraph::op::util::get_single_value(const_node, pad_value)) {
            IE_THROW() << "Unsupported pad value in " << op->get_friendly_name() << " (" << op->get_type_name() << ")";
        }
    }

    cldnn::border_type border_mode = GetBorderType(op->get_pad_mode());

    auto tilePrim = cldnn::border(layerName,
                                  inputPrimitives[0],
                                  pads_begin,
                                  pads_end,
                                  border_mode,
                                  pad_value);

    p.AddPrimitive(tilePrim);
    p.AddPrimitiveToProfiler(op);
}

REGISTER_FACTORY_IMPL(v1, Pad);

}  // namespace CLDNNPlugin
