// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "transformations/utils/utils.hpp"

#include "ngraph/op/one_hot.hpp"

#include "intel_gpu/primitives/one_hot.hpp"

namespace ov {
namespace intel_gpu {

static void CreateOneHotOp(Program& p, const std::shared_ptr<ngraph::op::v1::OneHot>& op) {
    p.ValidateInputs(op, {4});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    int64_t axis = op->get_axis();
    auto depth_value_node = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(op->get_input_node_shared_ptr(1));
    auto on_value_node = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(op->get_input_node_shared_ptr(2));
    auto off_value_node = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(op->get_input_node_shared_ptr(3));

    if (on_value_node == nullptr || off_value_node == nullptr || depth_value_node == nullptr)
        IE_THROW() << "Unsupported on/off/depth node type in " << op->get_friendly_name() << " (" << op->get_type_name() << ")";

    float on_value;
    float off_value;

    if (!ngraph::op::util::get_single_value(on_value_node, on_value) ||
        !ngraph::op::util::get_single_value(off_value_node, off_value)) {
        IE_THROW() << "Unsupported parameter size in " << op->get_friendly_name() << " (" << op->get_type_name() << ")";
    }

    auto dims = op->get_input_partial_shape(0);

    if (axis < -1 || axis > static_cast<int16_t>(dims.size()))
        IE_THROW() << op->get_friendly_name() << " Incorrect OneHot axis value: " << axis << ". Should be between -1 and " << dims.size();

    if (axis == -1) {
        axis = dims.size();
        for (int i = dims.size() - 1; i >= 0; i--) {
            if (dims[i] == 1)
                axis--;
            else
                break;
        }
    }

    int64_t depth = depth_value_node->cast_vector<int64_t>()[0];

    auto out_pshape = op->get_output_partial_shape(0);
    if (out_pshape.is_dynamic()) {
        IE_THROW() << "OneHot doesn't support dynamic shapes yet";
    }
    auto out_tensor = tensor_from_dims(out_pshape.to_shape());

    auto oneHotPrim = cldnn::one_hot(layerName,
                                     inputPrimitives[0],
                                     out_tensor,
                                     DataTypeFromPrecision(op->get_output_element_type(0)),
                                     axis,
                                     depth,
                                     on_value,
                                     off_value,
                                     op->get_friendly_name());

    p.AddPrimitive(oneHotPrim);
    p.AddPrimitiveToProfiler(op);
}

REGISTER_FACTORY_IMPL(v1, OneHot);

}  // namespace intel_gpu
}  // namespace ov
