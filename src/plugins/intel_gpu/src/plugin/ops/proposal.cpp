// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/proposal.hpp"

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/primitives/proposal.hpp"
#include "intel_gpu/primitives/mutable_data.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"

namespace ov {
namespace intel_gpu {

static void CreateProposalOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v0::Proposal>& op) {
    validate_inputs_count(op, {3});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    auto attrs = op->get_attrs();
    float nms_thresh = attrs.nms_thresh;
    int min_size = static_cast<int>(attrs.min_size);
    int feature_stride = static_cast<int>(attrs.feat_stride);
    int pre_nms_topn = static_cast<int>(attrs.pre_nms_topn);
    int post_nms_topn = static_cast<int>(attrs.post_nms_topn);
    const std::vector<float> ratio = attrs.ratio;
    const std::vector<float> scale = attrs.scale;
    float box_coordinate_scale = attrs.box_coordinate_scale;
    float box_size_scale = attrs.box_size_scale;
    int base_size = static_cast<int>(attrs.base_size);
    std::string framework = attrs.framework;
    bool normalize = attrs.normalize;
    bool clip_before_nms = attrs.clip_before_nms;
    bool clip_after_nms = attrs.clip_after_nms;

    float coordinates_offset;
    bool swap_xy;
    bool initial_clip;
    bool round_ratios;
    bool shift_anchors;

    if (framework == "tensorflow") {
        coordinates_offset = 0.0f;
        initial_clip = true;
        shift_anchors = true;
        round_ratios = false;
        swap_xy = true;
    } else {
        coordinates_offset = 1.0f;
        initial_clip = false;
        shift_anchors = false;
        round_ratios = true;
        swap_xy = false;
    }

    if (p.use_new_shape_infer()) {
        size_t num_outputs = op->get_output_size();

        auto proposalPrim = cldnn::proposal(layerName,
                                            inputs[0],  // cls_score
                                            inputs[1],  // bbox_pred
                                            inputs[2],  // im_info
                                            0,          // max_num_proposals is unused
                                            nms_thresh,
                                            base_size,
                                            min_size,
                                            feature_stride,
                                            pre_nms_topn,
                                            post_nms_topn,
                                            ratio,
                                            scale,
                                            coordinates_offset,
                                            box_coordinate_scale,
                                            box_size_scale,
                                            false,
                                            swap_xy,
                                            initial_clip,
                                            clip_before_nms,
                                            clip_after_nms,
                                            round_ratios,
                                            shift_anchors,
                                            normalize,
                                            cldnn::element_type_to_data_type(op->get_output_element_type(0)),
                                            num_outputs);
        proposalPrim.output_data_types = get_output_data_types(op);
        p.add_primitive(*op, proposalPrim);
    } else {
        if (op->get_output_size() == 2) {
            auto mutable_precision = op->get_output_element_type(1);
            if (mutable_precision == ov::element::i64) {
                mutable_precision = ov::element::i32;
            }

            cldnn::layout mutableLayout = cldnn::layout(cldnn::element_type_to_data_type(mutable_precision),
                                                        cldnn::format::get_default_format(op->get_output_shape(1).size()),
                                                        tensor_from_dims(op->get_output_shape(1)));

            GPU_DEBUG_LOG << "[" << layerName << ": mutable data]" << std::endl;
            auto shared_memory = p.get_engine().allocate_memory(mutableLayout);

            cldnn::primitive_id proposal_mutable_id_w = layerName + "_md_write";
            auto argmax_mutable_prim = cldnn::mutable_data(proposal_mutable_id_w,
                                                           shared_memory);
            p.add_primitive(*op, argmax_mutable_prim);
            inputs.push_back(cldnn::input_info(proposal_mutable_id_w));

            std::string proposalLayerName = layerName + ".out0";
            auto proposalPrim = cldnn::proposal(proposalLayerName,
                                                inputs[0],  // cls_score
                                                inputs[1],  // bbox_pred
                                                inputs[2],  // im_info
                                                inputs[3],  // second_output
                                                0,          // max_num_proposals is unused
                                                nms_thresh,
                                                base_size,
                                                min_size,
                                                feature_stride,
                                                pre_nms_topn,
                                                post_nms_topn,
                                                ratio,
                                                scale,
                                                coordinates_offset,
                                                box_coordinate_scale,
                                                box_size_scale,
                                                false,
                                                swap_xy,
                                                initial_clip,
                                                clip_before_nms,
                                                clip_after_nms,
                                                round_ratios,
                                                shift_anchors,
                                                normalize);

            p.add_primitive(*op, proposalPrim);

            cldnn::primitive_id proposal_mutable_id_r = layerName + ".out1";
            auto argmax_mutable_prim_r = cldnn::mutable_data(proposal_mutable_id_r,
                                                             { cldnn::input_info(proposalLayerName) },
                                                             shared_memory);
            p.add_primitive(*op, argmax_mutable_prim_r);
            return;
        } else if (op->get_output_size() == 1) {
            auto proposalPrim = cldnn::proposal(layerName,
                                                inputs[0],  // cls_score
                                                inputs[1],  // bbox_pred
                                                inputs[2],  // im_info
                                                0,          // max_num_proposals is unused
                                                nms_thresh,
                                                base_size,
                                                min_size,
                                                feature_stride,
                                                pre_nms_topn,
                                                post_nms_topn,
                                                ratio,
                                                scale,
                                                coordinates_offset,
                                                box_coordinate_scale,
                                                box_size_scale,
                                                false,
                                                swap_xy,
                                                initial_clip,
                                                clip_before_nms,
                                                clip_after_nms,
                                                round_ratios,
                                                shift_anchors,
                                                normalize);

            p.add_primitive(*op, proposalPrim);
        } else {
            OPENVINO_THROW(op->get_friendly_name(), " Incorrect Proposal outputs number");
        }
    }
}

REGISTER_FACTORY_IMPL(v0, Proposal);
REGISTER_FACTORY_IMPL(v4, Proposal);

}  // namespace intel_gpu
}  // namespace ov
