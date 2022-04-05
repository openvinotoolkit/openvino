// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "ngraph/op/result.hpp"

#include "intel_gpu/primitives/reorder.hpp"

using namespace InferenceEngine;

namespace ov {
namespace runtime {
namespace intel_gpu {

static void CreateResultOp(Program& p, const std::shared_ptr<ngraph::op::v0::Result>& op) {
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
        IE_THROW() << "Can't find output " << inputID << " in OutputsDataMap";
    }
    std::string originalOutName = it->first;
    DataPtr outputData = it->second;

    auto inputs = p.GetInputPrimitiveIDs(op);
    const auto outputDesc = outputData->getTensorDesc();
    auto outputlayout = outputDesc.getLayout();

    if (ngraph::is_type<ngraph::op::v8::NV12toRGB>(prev) ||
        ngraph::is_type<ngraph::op::v8::NV12toBGR>(prev) ||
        ngraph::is_type<ngraph::op::v8::I420toRGB>(prev) ||
        ngraph::is_type<ngraph::op::v8::I420toBGR>(prev)) {
        outputlayout = NHWC;
    }

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
        IE_THROW() << "Unsupported layout (" << outputlayout << ") in output: " << originalOutName;
    }

    auto outLayerName = layer_type_name_ID(op);
    Precision precision = outputData->getPrecision();
    std::string outputID = inputs[0];

    p.AddPrimitive(cldnn::reorder(outLayerName,
                                  outputID,
                                  FormatFromLayout(outputlayout),
                                  DataTypeFromPrecision(precision),
                                  std::vector<float>(),
                                  cldnn::reorder_mean_mode::subtract,
                                  op->get_friendly_name()));
    p.InitProfileInfo(outLayerName, "reorder");
    p.profilingIDs.push_back(outLayerName);
    p.primitiveIDs[outLayerName] = outLayerName;
    p.primitiveIDs[originalOutName] = outLayerName;

    p.outputDims[originalOutName] = outputDesc.getDims();
    p.prevPrimitiveIDs[outLayerName] = {originalOutName};
}

REGISTER_FACTORY_IMPL(v0, Result);

}  // namespace intel_gpu
}  // namespace runtime
}  // namespace ov
