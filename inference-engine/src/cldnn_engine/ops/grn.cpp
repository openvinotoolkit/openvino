// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_program.h"
#include "cldnn_common_utils.h"

#include "ngraph/op/grn.hpp"

#include "api/grn.hpp"

namespace CLDNNPlugin {

void CreateGRNOp(Program& p, const std::shared_ptr<ngraph::op::v0::GRN>& op) {
    p.ValidateInputs(op, {1});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    auto primitive = cldnn::grn(layerName,
                                inputPrimitives[0],
                                op->get_bias(),
                                DataTypeFromPrecision(op->get_output_element_type(0)));

    p.AddPrimitive(primitive);
    p.AddPrimitiveToProfiler(op);
}

REGISTER_FACTORY_IMPL(v0, GRN);

}  // namespace CLDNNPlugin
