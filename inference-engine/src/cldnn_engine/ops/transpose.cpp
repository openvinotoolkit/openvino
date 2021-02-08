// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_program.h"
#include "cldnn_common_utils.h"

#include "ngraph/op/transpose.hpp"
#include "ngraph/op/constant.hpp"

#include "api/permute.hpp"

namespace CLDNNPlugin {

void CreateTransposeOp(Program& p, const std::shared_ptr<ngraph::op::v1::Transpose>& op) {
    p.ValidateInputs(op, {1, 2});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    std::vector<uint16_t> ie_order;
    if (op->get_input_size() == 2) {
        auto order_constant = std::dynamic_pointer_cast<ngraph::op::Constant>(op->get_input_node_shared_ptr(1));
        if (!order_constant) {
            THROW_IE_EXCEPTION << "Unsupported parameter nodes type in " << op->get_friendly_name() << " (" << op->get_type_name() << ")";
        }
        ie_order = order_constant->cast_vector<uint16_t>();
    }

    int rank = std::max(4, static_cast<int>(op->get_input_shape(0).size()));
    if (ie_order.empty()) {
        // if order size is less than 4 - fill the rest with just copy
        for (int o = rank - 1; o >= 0; o--)
            ie_order.push_back((uint16_t)o);
    }

    std::vector<uint16_t> cldnn_permute_order = ConvertPermuteOrder(ie_order, rank);

    auto permutePrim = cldnn::permute(layerName,
                                      inputPrimitives[0],
                                      cldnn_permute_order);

    p.AddPrimitive(permutePrim);
    p.AddPrimitiveToProfiler(op);
}

REGISTER_FACTORY_IMPL(v1, Transpose);

}  // namespace CLDNNPlugin
