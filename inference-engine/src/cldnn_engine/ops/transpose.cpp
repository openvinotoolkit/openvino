// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_program.h"

#include "ngraph/op/transpose.hpp"
#include "ngraph/op/constant.hpp"

#include "api/permute.hpp"

namespace CLDNNPlugin {

template<class Type>
std::vector<Type> GetPermuteOrder(const std::vector<Type>& ie_order, Type value_to_align = 0) {
    static_assert(std::is_integral<Type>::value, "Integeral required.");
    std::vector<Type> cldnn_order = ie_order;

    // 1. Align to min. 4 sizes
    if (cldnn_order.size() < 4)
        cldnn_order.push_back(value_to_align);

    // 2. Swap spatial positions
    for (int i = 0; i < (cldnn_order.size() - 2) / 2; i++) {
        std::swap(cldnn_order[2 + i], cldnn_order[1 + cldnn_order.size() - (2 + i)]);
    }

    return cldnn_order;
}

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

    // if order size is less than 4 - fill the rest with just copy
    for (auto o = ie_order.size(); o < rank; o++)
        ie_order.push_back((uint16_t)o);

    /*
        Because of the cldnn ordering: bfxy, and IE ordering: bfyx
        we need to adjust the permute order.
    */
    std::vector<uint16_t> cldnn_permute_order;
    // 1. Switch permute order values for spatial dims
    for (auto const& o : ie_order) {
        if (o >= 2)
            cldnn_permute_order.push_back(1 + ie_order.size() - o);
        else
            cldnn_permute_order.push_back(o);
    }
    cldnn_permute_order = GetPermuteOrder(cldnn_permute_order);

    auto permutePrim = cldnn::permute(layerName,
                                      inputPrimitives[0],
                                      cldnn_permute_order);

    p.AddPrimitive(permutePrim);
    p.AddPrimitiveToProfiler(op);
}

REGISTER_FACTORY_IMPL(v1, Transpose);

}  // namespace CLDNNPlugin
