// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_program.h"
#include "cldnn_common_utils.h"

#include "ngraph/op/result.hpp"

#include "api/reorder.hpp"

using namespace InferenceEngine;

namespace CLDNNPlugin {

void CreateResultOp(Program& p, const std::shared_ptr<ngraph::op::v0::Result>& op) {
    OutputsDataMap networkOutputs = p.GetNetworkOutputs();
    p.ValidateInputs(op, {1});

    auto prev = op->get_input_node_shared_ptr(0);
    NGRAPH_SUPPRESS_DEPRECATED_START
    auto inputID = op->get_input_source_output(0).get_tensor().get_name();
    NGRAPH_SUPPRESS_DEPRECATED_END
    if (inputID.empty()) {
        inputID = prev->get_friendly_name();
        if (prev->get_output_size() > 1) {
            inputID += "." + std::to_string(op->get_input_source_output(0).get_index());
        }
    }
    auto it = networkOutputs.find(inputID);
    if (it == networkOutputs.end()) {
        THROW_IE_EXCEPTION << "Can't find output " << inputID << " in OutputsDataMap";
    }
    std::string originalOutName = it->first;
    DataPtr outputData = it->second;

    auto inputs = p.GetInputPrimitiveIDs(op);
    const auto outputDesc = outputData->getTensorDesc();
    const auto outputlayout = outputDesc.getLayout();

    // TODO: add precision check once there's an outputInfo object
    if (outputlayout != NCHW &&
        // TODO: change 6d case once new layout added in IE
        outputlayout != BLOCKED &&
        outputlayout != NCDHW &&
        outputlayout != NHWC &&
        outputlayout != CHW &&
        outputlayout != NC &&
        outputlayout != C &&
        outputlayout != SCALAR) {
        THROW_IE_EXCEPTION << "Unsupported layout (" << outputlayout << ") in output: " << originalOutName;
    }

    auto outLayerName = layer_type_name_ID(op);
    Precision precision = outputData->getPrecision();
    std::string outputID = inputs[0];

    p.AddPrimitive(cldnn::reorder(outLayerName,
                                outputID,
                                FormatFromLayout(outputData->getLayout()),
                                DataTypeFromPrecision(precision)));
    p.InitProfileInfo(outLayerName, "reorder");
    p.profilingIDs.push_back(outLayerName);
    p.primitiveIDs[outLayerName] = outLayerName;
    p.primitiveIDs[originalOutName] = outLayerName;

    p.outputDims[originalOutName] = outputDesc.getDims();
    p.prevPrimitiveIDs[outLayerName] = {originalOutName};
}

REGISTER_FACTORY_IMPL(v0, Result);

}  // namespace CLDNNPlugin
