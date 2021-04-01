// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_program.h"
#include "cldnn_common_utils.h"
#include "transformations/utils/utils.hpp"

#include "ngraph/op/one_hot.hpp"

#include "api/one_hot.hpp"

namespace CLDNNPlugin {

void CreateOneHotOp(Program& p, const std::shared_ptr<ngraph::op::v1::OneHot>& op) {
    p.ValidateInputs(op, {4});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    int16_t axis = op->get_axis();
    auto on_value_node = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(op->get_input_node_shared_ptr(2));
    auto off_value_node = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(op->get_input_node_shared_ptr(3));

    if (on_value_node == nullptr || off_value_node == nullptr)
        THROW_IE_EXCEPTION << "Unsupported on/off node type in " << op->get_friendly_name() << " (" << op->get_type_name() << ")";

    float on_value;
    float off_value;

    if (!ngraph::op::util::get_single_value(on_value_node, on_value) ||
        !ngraph::op::util::get_single_value(off_value_node, off_value)) {
        THROW_IE_EXCEPTION << "Unsupported parameter size in " << op->get_friendly_name() << " (" << op->get_type_name() << ")";
    }

    auto dims = op->get_input_shape(0);

    if (axis < -1 || axis > static_cast<int16_t>(dims.size()))
        THROW_IE_EXCEPTION << op->get_friendly_name() << " Incorrect OneHot axis value: " << axis << ". Should be between -1 and " << dims.size();

    if (axis == -1) {
        axis = dims.size();
        for (int i = dims.size() - 1; i >= 0; i--) {
            if (dims[i] == 1)
                axis--;
            else
                break;
        }
    }

    auto oneHotPrim = cldnn::one_hot(layerName,
                                     inputPrimitives[0],
                                     CldnnTensorFromIEDims(op->get_output_shape(0)),
                                     DataTypeFromPrecision(op->get_output_element_type(0)),
                                     static_cast<uint16_t>(axis),
                                     on_value,
                                     off_value);

    p.AddPrimitive(oneHotPrim);
    p.AddPrimitiveToProfiler(op);
}

REGISTER_FACTORY_IMPL(v1, OneHot);

}  // namespace CLDNNPlugin
