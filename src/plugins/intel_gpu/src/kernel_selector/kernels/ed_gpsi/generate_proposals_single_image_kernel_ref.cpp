// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "generate_proposals_single_image_kernel_ref.h"
#include "kernel_selector_utils.h"
#include <algorithm>
#include <string>

namespace kernel_selector {

ParamsKey ExperimentalDetectronGenerateProposalsSingleImageRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableAllInputLayout();
    k.EnableAllOutputLayout();
    k.EnableBatching();
    k.EnableDifferentTypes();
    k.EnableTensorPitches();
    return k;
}

KernelsPriority ExperimentalDetectronGenerateProposalsSingleImageRef::GetKernelsPriority(const Params&) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}

bool ExperimentalDetectronGenerateProposalsSingleImageRef::Validate(const Params& p) const {
    if (p.GetType() != KernelType::EXPERIMENTAL_DETECTRON_GENERATE_PROPOSALS_SINGLE_IMAGE) {
        return false;
    }
    return true;
}

namespace {
constexpr size_t kImInfoInputIdx = 0;
constexpr size_t kAnchorsInputIdx = 1;
constexpr size_t kDeltasInputIdx = 2;
constexpr size_t kScoresInputIdx = 3;

ExperimentalDetectronGenerateProposalsSingleImageRef::DispatchData SetDefault(
        const experimental_detectron_generate_proposals_single_image_params& params, size_t idx) {
    ExperimentalDetectronGenerateProposalsSingleImageRef::DispatchData dispatch_data;

    if (idx == 0) {
        const auto bottom_H = params.inputs[kDeltasInputIdx].Feature().v;
        const auto bottom_W = params.inputs[kDeltasInputIdx].Y().v;
        const auto anchors_num = params.inputs[kScoresInputIdx].Batch().v;
        dispatch_data.gws = {bottom_H, bottom_W, anchors_num};
    } else if (idx == 3) {
        dispatch_data.gws = {params.post_nms_count, 1, 1};
    } else {
        dispatch_data.gws = {1, 1, 1};
    }
    dispatch_data.lws = GetOptimalLocalWorkGroupSizes(dispatch_data.gws, params.engineInfo);

    return dispatch_data;
}
}  // namespace

void ExperimentalDetectronGenerateProposalsSingleImageRef::SetKernelArguments(
        const experimental_detectron_generate_proposals_single_image_params& params,
        size_t idx, cldnn::arguments_desc& arguments) const {
    switch (idx) {
        case 0: { // refine anchors
            arguments.push_back({ArgumentDescriptor::Types::INPUT, kImInfoInputIdx});
            arguments.push_back({ArgumentDescriptor::Types::INPUT, kAnchorsInputIdx});
            arguments.push_back({ArgumentDescriptor::Types::INPUT, kDeltasInputIdx});
            arguments.push_back({ArgumentDescriptor::Types::INPUT, kScoresInputIdx});
            arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0}); // proposals
            break;
        }
        case 1: { // sort proposals by score
            arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0}); // proposals
            break;
        }
        case 2: { // NMS
            arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0}); // proposals
            arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1}); // nms_out_indices
            arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 2}); // nms_num_outputs
            break;
        }
        case 3: { // convert proposals to rois and roi scores
            arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0}); // proposals
            arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1}); // nms_out_indices
            arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 2}); // nms_num_outputs
            arguments.push_back({ArgumentDescriptor::Types::OUTPUT, 0});          // rois
            arguments.push_back({ArgumentDescriptor::Types::OUTPUT, 1}); // roi scores
            break;
        }
        default:
            throw std::invalid_argument("experimental_detectron_generate_proposals_single_image has 4 kernels. valid index is 0 ~ 3.");
    }
}

KernelsData ExperimentalDetectronGenerateProposalsSingleImageRef::GetKernelsData(const Params& params) const {
    if (!Validate(params)) {
        return {};
    }

    constexpr size_t kKernelsNum = 4;
    KernelData kd = KernelData::Default<experimental_detectron_generate_proposals_single_image_params>(params, kKernelsNum);
    const experimental_detectron_generate_proposals_single_image_params& new_params
        = static_cast<const experimental_detectron_generate_proposals_single_image_params&>(params);

    const auto anchors_num = new_params.inputs[kScoresInputIdx].Batch().v;
    const auto bottom_H = new_params.inputs[kDeltasInputIdx].Feature().v;
    const auto bottom_W = new_params.inputs[kDeltasInputIdx].Y().v;
    const auto num_proposals = anchors_num * bottom_H * bottom_W;
    const auto pre_nms_topn = std::min(num_proposals, new_params.pre_nms_count);
    const auto max_delta_log_wh = static_cast<float>(std::log(1000.0 / 16.0));
    kd.internalBufferDataType = Datatype::F32;

    constexpr size_t kProposalBoxSize = 5; // 5 values: {x0, y0, x1, y1, score}
    const auto proposals_buffer_size = num_proposals * sizeof(float) * kProposalBoxSize;
    kd.internalBufferSizes.push_back(proposals_buffer_size);

    const auto out_indices_size = new_params.post_nms_count * sizeof(float);
    kd.internalBufferSizes.push_back(out_indices_size);

    kd.internalBufferSizes.push_back(sizeof(size_t));

    for (size_t i = 0; i < kKernelsNum; ++i) {
        const auto dispatchData = SetDefault(new_params, i);
        const auto entry_point = GetEntryPoint(kernelName, new_params.layerID, params, i);
        auto cldnn_jit = MakeBaseParamsJitConstants(new_params);


        cldnn_jit.AddConstant(MakeJitConstant("EDGPSI_STAGE_" + std::to_string(i), "true"));
        switch (i) {
            case 0: {
                cldnn_jit.AddConstants({MakeJitConstant("MIN_SIZE", new_params.min_size),
                                        MakeJitConstant("ANCHORS_NUM", anchors_num),
                                        MakeJitConstant("BOTTOM_H", bottom_H),
                                        MakeJitConstant("BOTTOM_W", bottom_W),
                                        MakeJitConstant("BOTTOM_AREA", bottom_H * bottom_W),
                                        MakeJitConstant("MAX_DELTA_LOG_WH", max_delta_log_wh)
                });
                break;
            }
            case 1: {
                cldnn_jit.AddConstants({MakeJitConstant("NUM_PROPOSALS", num_proposals),
                                        MakeJitConstant("PRE_NMS_TOPN", pre_nms_topn)
                });
                break;
            }
            case 2: {
                cldnn_jit.AddConstants({MakeJitConstant("PRE_NMS_TOPN", pre_nms_topn),
                                        MakeJitConstant("POST_NMS_COUNT", new_params.post_nms_count),
                                        MakeJitConstant("NMS_THRESHOLD", new_params.nms_threshold)
                                       });
                break;
            }
            case 3: {
                cldnn_jit.AddConstants({MakeJitConstant("POST_NMS_COUNT", new_params.post_nms_count)});
                break;
            }
            default:
                throw std::invalid_argument("EDGPSI has 4 kernels. valid index is 0 ~ 3.");
        }

        const auto jit = CreateJit(kernelName, cldnn_jit, entry_point);
        auto& kernel = kd.kernels[i];

        KernelBase::CheckDispatchData(kernelName, dispatchData, params.engineInfo.maxWorkGroupSize);
        kernel.params.workGroups.global = dispatchData.gws;
        kernel.params.workGroups.local  = dispatchData.lws;
        kernel.code.kernelString = GetKernelString(kernelName, jit, entry_point, params.engineInfo);
        SetKernelArguments(new_params, i, kernel.params.arguments);
    }

    return {kd};
}
}  // namespace kernel_selector
