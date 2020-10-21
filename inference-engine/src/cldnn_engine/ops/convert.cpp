// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_program.h"
#include "cldnn_common_utils.h"

#include "ngraph/op/convert.hpp"
#include "ngraph/op/convert_like.hpp"

#include "api/reorder.hpp"

namespace CLDNNPlugin {

void CreateConvertLikeOp(Program& p, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v1::ConvertLike>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    p.ValidateInputs(op, {2});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    auto outDataType = DataTypeFromPrecision(op->get_input_element_type(1));

    auto reorderPrim = cldnn::reorder(layerName, inputPrimitives[0], cldnn::format::any, outDataType);

    p.AddPrimitive(reorderPrim);
    p.AddPrimitiveToProfiler(op);
}

void CreateConvertOp(Program& p, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v0::Convert>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    p.ValidateInputs(op, {1});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    auto outDataType = DataTypeFromPrecision(op->get_destination_type());

    auto reorderPrim = cldnn::reorder(layerName, inputPrimitives[0], cldnn::format::any, outDataType);

    p.AddPrimitive(reorderPrim);
    p.AddPrimitiveToProfiler(op);
}

REGISTER_FACTORY_IMPL(v0, Convert);
REGISTER_FACTORY_IMPL(v1, ConvertLike);

}  // namespace CLDNNPlugin
