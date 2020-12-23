// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_program.h"
#include "cldnn_common_utils.h"

#include "ngraph/op/reshape.hpp"
#include "ngraph/op/squeeze.hpp"
#include "ngraph/op/unsqueeze.hpp"

#include "api/reshape.hpp"
#include "api/reorder.hpp"

namespace CLDNNPlugin {

void CreateCommonReshapeOp(Program& p, const std::shared_ptr<ngraph::Node>& op) {
    p.ValidateInputs(op, {1, 2});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    auto inDims = op->get_input_shape(0);
    auto outDims = op->get_output_shape(0);
    auto outTensor = CldnnTensorFromIEDims(outDims);

    // if we convert from or to 5D/6D, additional reorder also required to change format
    cldnn::primitive_id reshapeInputId = inputPrimitives[0];
    if (inDims.size() != outDims.size()) {
        cldnn::primitive_id reorderId = "reorder:" + op->get_friendly_name() + "_reorder";
        cldnn::format outputFormat = cldnn::format::bfyx;

        switch (outDims.size()) {
        case 5: outputFormat = cldnn::format::bfzyx; break;
        case 6: outputFormat = cldnn::format::bfwzyx; break;
        default: break;
        }

        cldnn::layout outputLayout(DataTypeFromPrecision(op->get_output_element_type(0)), outputFormat, outTensor);
        p.AddPrimitive(cldnn::reorder(reorderId, reshapeInputId, outputLayout));
        p.InitProfileInfo(reorderId, "Reorder", false, InferenceEngine::InferenceEngineProfileInfo::EXECUTED, layerName);
        p.primitivesToIRLayersMap[reorderId] = { op->get_friendly_name() };
        p.primitiveIDs[layerName + "_reorder"] = reorderId;
        p.primitiveIDs[reorderId] = reorderId;
        p.profilingIDs.push_back(reorderId);
        reshapeInputId = reorderId;
    }

    auto reshapePrim = cldnn::reshape(layerName,
                                      reshapeInputId,
                                      outTensor);

    p.AddPrimitive(reshapePrim);
    p.AddPrimitiveToProfiler(op);
}

void CreateReshapeOp(Program& p, const std::shared_ptr<ngraph::op::v1::Reshape>& op) {
    CreateCommonReshapeOp(p, op);
}

void CreateSqueezeOp(Program& p, const std::shared_ptr<ngraph::op::v0::Squeeze>& op) {
    CreateCommonReshapeOp(p, op);
}

void CreateUnsqueezeOp(Program& p, const std::shared_ptr<ngraph::op::v0::Unsqueeze>& op) {
    CreateCommonReshapeOp(p, op);
}

REGISTER_FACTORY_IMPL(v1, Reshape);
REGISTER_FACTORY_IMPL(v0, Squeeze);
REGISTER_FACTORY_IMPL(v0, Unsqueeze);

}  // namespace CLDNNPlugin
