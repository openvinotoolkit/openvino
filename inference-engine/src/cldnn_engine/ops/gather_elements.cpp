// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_program.h"
#include "cldnn_common_utils.h"

#include "ngraph/op/gather_elements.hpp"
#include "ngraph/op/constant.hpp"

#include "api/gather_elements.hpp"

namespace CLDNNPlugin {

void CreateGatherElementsOp(Program& p, const std::shared_ptr<ngraph::op::v6::GatherElements>& op) {
    p.ValidateInputs(op, {2});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    auto axis = op->get_axis();
    auto outLayout = DefaultFormatForDims(op->get_output_shape(0).size());

    auto primitive = cldnn::gather_elements(layerName,
                                           inputPrimitives[0],
                                           inputPrimitives[1],
                                           outLayout,
                                           CldnnTensorFromIEDims(op->get_output_shape(0)),
                                           axis);

    p.AddPrimitive(primitive);
    p.AddPrimitiveToProfiler(op);
}

REGISTER_FACTORY_IMPL(v6, GatherElements);

}  // namespace CLDNNPlugin