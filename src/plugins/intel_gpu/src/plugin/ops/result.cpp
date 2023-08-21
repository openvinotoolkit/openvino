// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/result.hpp"
#include "openvino/op/nv12_to_rgb.hpp"
#include "openvino/op/nv12_to_bgr.hpp"
#include "openvino/op/i420_to_rgb.hpp"
#include "openvino/op/i420_to_bgr.hpp"

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/primitives/reorder.hpp"

using namespace InferenceEngine;

namespace ov {
namespace intel_gpu {

static void CreateResultOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v0::Result>& op) {
    OutputsDataMap networkOutputs = p.GetNetworkOutputs();
    validate_inputs_count(op, {1});

    auto prev = op->get_input_node_shared_ptr(0);
    NGRAPH_SUPPRESS_DEPRECATED_START
    auto inputID = ov::descriptor::get_ov_tensor_legacy_name(op->get_input_source_output(0).get_tensor());
    NGRAPH_SUPPRESS_DEPRECATED_END
    if (inputID.empty()) {
        inputID = prev->get_friendly_name();
        if (prev->get_output_size() > 1) {
            inputID += "." + std::to_string(op->get_input_source_output(0).get_index());
        }
    }
    auto it = networkOutputs.find(inputID);
    OPENVINO_ASSERT(it != networkOutputs.end(), "[GPU] Can't find output ", inputID, " in OutputsDataMap");
    std::string originalOutName = it->first;
    DataPtr outputData = it->second;

    auto inputs = p.GetInputInfo(op);
    const auto outputDesc = outputData->getTensorDesc();
    auto outputlayout = outputDesc.getLayout();

    if (ov::is_type<ov::op::v8::NV12toRGB>(prev) ||
        ov::is_type<ov::op::v8::NV12toBGR>(prev) ||
        ov::is_type<ov::op::v8::I420toRGB>(prev) ||
        ov::is_type<ov::op::v8::I420toBGR>(prev)) {
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
        OPENVINO_THROW("[GPU] Unsupported layout (", outputlayout, ") in output: ", originalOutName);
    }
    auto out_rank = op->get_output_partial_shape(0).size();
    auto out_format = cldnn::format::get_default_format(out_rank);
    std::vector<size_t> default_order(out_rank);
    std::iota(default_order.begin(), default_order.end(), 0);
    // For legacy API we need to handle NHWC as well, so check non default order
    if (outputlayout == NHWC) {
        out_format = FormatFromLayout(outputlayout);
    }

    auto outLayerName = layer_type_name_ID(op);
    auto inputDataType = cldnn::element_type_to_data_type(op->get_input_element_type(0));
    Precision precision = outputData->getPrecision();
    auto outputDataType = DataTypeFromPrecision(precision);
    cldnn::input_info outputID = inputs[0];

    // Even for result op, if reorder only performs type conversion, reorder is created in truncation mode
    if (inputDataType != outputDataType) {
        auto reorder_primitive = cldnn::reorder(outLayerName,
                                                outputID,
                                                out_format,
                                                outputDataType,
                                                std::vector<float>(),
                                                cldnn::reorder_mean_mode::subtract,
                                                cldnn::padding(),
                                                true);
        p.add_primitive(*op, reorder_primitive, {originalOutName});
    } else {
        auto reorder_primitive = cldnn::reorder(outLayerName,
                                                outputID,
                                                out_format,
                                                outputDataType);
        p.add_primitive(*op, reorder_primitive, {originalOutName});
    }
    p.outputDims[originalOutName] = outputDesc.getDims();
    p.prevPrimitiveIDs[outLayerName] = {originalOutName};
}

REGISTER_FACTORY_IMPL(v0, Result);

}  // namespace intel_gpu
}  // namespace ov
