// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_program.h"
#include "cldnn_common_utils.h"

#include "ngraph/op/ctc_greedy_decoder.hpp"

#include "api/ctc_greedy_decoder.hpp"

namespace CLDNNPlugin {

void CreateCTCGreedyDecoderOp(Program& p, const std::shared_ptr<ngraph::op::v0::CTCGreedyDecoder>& op) {
    p.ValidateInputs(op, {2});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    auto primitive = cldnn::ctc_greedy_decoder(layerName,
                                               inputPrimitives[0],
                                               inputPrimitives[1],
                                               op->get_ctc_merge_repeated(),
                                               DataTypeFromPrecision(op->get_output_element_type(0)),
                                               CldnnTensorFromIEDims(op->get_output_shape(0)));

    p.AddPrimitive(primitive);
    p.AddPrimitiveToProfiler(op);
}

REGISTER_FACTORY_IMPL(v0, CTCGreedyDecoder);

}  // namespace CLDNNPlugin
