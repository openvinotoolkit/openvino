// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_program.h"
#include "cldnn_common_utils.h"

#include "ngraph/op/detection_output.hpp"

#include "api/detection_output.hpp"

namespace CLDNNPlugin {

static cldnn::prior_box_code_type PriorBoxCodeFromString(const std::string& str) {
    static const std::map<std::string, cldnn::prior_box_code_type> CodeNameToType = {
        { "caffe.PriorBoxParameter.CORNER" , cldnn::prior_box_code_type::corner },
        { "caffe.PriorBoxParameter.CENTER_SIZE" , cldnn::prior_box_code_type::center_size },
        { "caffe.PriorBoxParameter.CORNER_SIZE" , cldnn::prior_box_code_type::corner_size },
    };
    auto it = CodeNameToType.find(str);
    if (it != CodeNameToType.end()) {
        return it->second;
    } else {
        THROW_IE_EXCEPTION << "Unknown Prior-Box code type: " << str;
    }
    return cldnn::prior_box_code_type::corner;
}

void CreateDetectionOutputOp(Program& p, const std::shared_ptr<ngraph::op::v0::DetectionOutput>& op) {
    p.ValidateInputs(op, {3});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    auto attrs = op->get_attrs();

    uint32_t num_classes            = attrs.num_classes;
    bool share_location             = attrs.share_location;
    int background_label_id         = attrs.background_label_id;
    float nms_threshold             = attrs.nms_threshold;
    int top_k                       = attrs.top_k;
    float confidence_threshold      = attrs.confidence_threshold;
    float eta                       = 1.0f;
    int keep_top_k                  = attrs.keep_top_k[0];
    bool variance_encoded_in_target = attrs.variance_encoded_in_target;
    int input_width                 = attrs.input_width;
    int input_height                = attrs.input_height;
    bool normalized                 = attrs.normalized;
    std::string code_type           = attrs.code_type;
    bool clip_before_nms            = attrs.clip_before_nms;
    bool clip_after_nms             = attrs.clip_after_nms;
    bool decrease_label_id          = attrs.decrease_label_id;

    cldnn::prior_box_code_type cldnnCodeType = PriorBoxCodeFromString(code_type);
    int32_t prior_info_size = normalized != 0 ? 4 : 5;
    int32_t prior_coordinates_offset = normalized != 0 ? 0 : 1;

    auto detectionPrim = cldnn::detection_output(layerName,
                                                 inputPrimitives[0],
                                                 inputPrimitives[1],
                                                 inputPrimitives[2],
                                                 num_classes,
                                                 keep_top_k,
                                                 share_location,
                                                 background_label_id,
                                                 nms_threshold,
                                                 top_k,
                                                 eta,
                                                 cldnnCodeType,
                                                 variance_encoded_in_target,
                                                 confidence_threshold,
                                                 prior_info_size,
                                                 prior_coordinates_offset,
                                                 normalized,
                                                 input_width,
                                                 input_height,
                                                 decrease_label_id,
                                                 clip_before_nms,
                                                 clip_after_nms);

    p.AddPrimitive(detectionPrim);
    p.AddPrimitiveToProfiler(op);
}

REGISTER_FACTORY_IMPL(v0, DetectionOutput);

}  // namespace CLDNNPlugin
