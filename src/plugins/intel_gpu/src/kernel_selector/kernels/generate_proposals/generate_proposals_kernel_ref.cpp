// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "generate_proposals_kernel_ref.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {

ParamsKey GenerateProposalsRef::GetSupportedKey() const {
    ParamsKey k;

    k.EnableInputDataType(Datatype::INT32);
    k.EnableInputDataType(Datatype::INT64);
    k.EnableOutputDataType(Datatype::INT32);
    k.EnableOutputDataType(Datatype::INT64);
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

KernelsPriority GenerateProposalsRef::GetKernelsPriority(const Params&) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}

bool GenerateProposalsRef::Validate(const Params& p) const {
    if (p.GetType() != KernelType::GENERATE_PROPOSALS) {
        return false;
    }

    return true;
}

namespace {
constexpr size_t kImInfoInputIdx = 0;
constexpr size_t kAnchorsInputIdx = 1;
constexpr size_t kDeltasInputIdx = 2;
constexpr size_t kScoresInputIdx = 3;

GenerateProposalsRef::DispatchData SetDefault(const generate_proposals_params& params, size_t idx) {
    GenerateProposalsRef::DispatchData dispatch_data;

    const auto& inputs = params.inputs;
    const auto num_batches = inputs[kScoresInputIdx].Batch().v;
    const auto anchors_num = inputs[kScoresInputIdx].Feature().v;
    const auto bottom_H = inputs[kDeltasInputIdx].Y().v;
    const auto bottom_W = inputs[kDeltasInputIdx].X().v;

    if (idx == 0) {
        dispatch_data.gws = {bottom_H, bottom_W, num_batches * anchors_num};
    } else if (idx == 1 || idx == 2) {
        dispatch_data.gws = {num_batches, 1, 1};
    } else if (idx == 3) {
        dispatch_data.gws = {1, 1, 1};
    }

    dispatch_data.lws = GetOptimalLocalWorkGroupSizes(dispatch_data.gws, params.engineInfo);

    return dispatch_data;
}
}  // namespace

void GenerateProposalsRef::SetKernelArguments(
        const generate_proposals_params& params,
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
            arguments.push_back({ArgumentDescriptor::Types::OUTPUT, 2}); // rois num
            break;
        }
        case 3: { // convert proposals to rois and roi scores
            arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0}); // proposals
            arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1}); // nms_out_indices
            arguments.push_back({ArgumentDescriptor::Types::OUTPUT, 2}); // rois num
            arguments.push_back({ArgumentDescriptor::Types::OUTPUT, 0}); // rois
            arguments.push_back({ArgumentDescriptor::Types::OUTPUT, 1}); // roi scores
            break;
        }
        default:
            throw std::invalid_argument("generate_proposals has 4 kernels. valid index is 0 ~ 3.");
    }
}

KernelsData GenerateProposalsRef::GetKernelsData(const Params& params) const {
    if (!Validate(params)) {
        return {};
    }

    constexpr size_t kKernelsNum = 4;
    KernelData kd = KernelData::Default<generate_proposals_params>(params, kKernelsNum);
    const generate_proposals_params& new_params = static_cast<const generate_proposals_params&>(params);
    const auto& inputs = new_params.inputs;

    const auto anchors_num = inputs[kScoresInputIdx].Feature().v;
    const auto bottom_H = inputs[kDeltasInputIdx].Y().v;
    const auto bottom_W = inputs[kDeltasInputIdx].X().v;
    const auto scale_w_index = inputs[kImInfoInputIdx].Feature().v == 3 ? 2 : 3;
    const auto num_proposals = anchors_num * bottom_H * bottom_W;
    const auto pre_nms_topn = std::min(num_proposals, new_params.pre_nms_count);
    const auto max_delta_log_wh = static_cast<float>(std::log(1000.0 / 16.0));

    kd.internalBufferDataType = Datatype::F32;

    const auto num_batches = inputs[kScoresInputIdx].Batch().v;
    constexpr size_t kProposalBoxSize = 6; // 6 values: {x0, y0, x1, y1, score, keep}
    const auto proposals_buffer_size = num_batches * num_proposals * sizeof(float) * kProposalBoxSize;
    kd.internalBuffers.push_back(proposals_buffer_size);

    const auto out_indices_size = num_batches * new_params.post_nms_count * sizeof(float);
    kd.internalBuffers.push_back(out_indices_size);

    for (size_t i = 0; i < kKernelsNum; ++i) {
        const auto dispatchData = SetDefault(new_params, i);
        const auto entry_point = GetEntryPoint(kernelName, new_params.layerID, params, i);
        auto cldnn_jit = MakeBaseParamsJitConstants(new_params);

        cldnn_jit.AddConstant(MakeJitConstant("GENERATE_PROPOSALS_STAGE_" + std::to_string(i), "true"));
        cldnn_jit.AddConstant(MakeJitConstant("PROPOSAL_SIZE", kProposalBoxSize));
        cldnn_jit.Merge(MakeTypeJitConstants(new_params.roi_num_type, "ROI_NUM"));
        if (new_params.normalized) {
            cldnn_jit.AddConstant(MakeJitConstant("NORMALIZED", 1));
        }

        switch (i) {
            case 0: {
                cldnn_jit.AddConstants({MakeJitConstant("MIN_SIZE", new_params.min_size),
                                        MakeJitConstant("ANCHORS_NUM", anchors_num),
                                        MakeJitConstant("NUM_PROPOSALS", num_proposals),
                                        MakeJitConstant("BOTTOM_H", bottom_H),
                                        MakeJitConstant("BOTTOM_W", bottom_W),
                                        MakeJitConstant("BOTTOM_AREA", bottom_H * bottom_W),
                                        MakeJitConstant("SCALE_W_INDEX", scale_w_index),
                                        MakeJitConstant("MAX_DELTA_LOG_WH", max_delta_log_wh)
                                       });
                break;
            }
            case 1: {
                cldnn_jit.AddConstants({MakeJitConstant("NUM_PROPOSALS", num_proposals)});
                break;
            }
            case 2: {
                cldnn_jit.AddConstants({MakeJitConstant("NUM_PROPOSALS", num_proposals),
                                        MakeJitConstant("PRE_NMS_TOPN", pre_nms_topn),
                                        MakeJitConstant("POST_NMS_COUNT", new_params.post_nms_count),
                                        MakeJitConstant("NMS_THRESHOLD", new_params.nms_threshold),
                                       });
                break;
            }
            case 3: {
                cldnn_jit.AddConstants({MakeJitConstant("POST_NMS_COUNT", new_params.post_nms_count),
                                        MakeJitConstant("NUM_PROPOSALS", num_proposals)
                                       });
                break;
            }
            default:
                throw std::invalid_argument("GENERATE_PROPOSALS has 4 kernels. valid index is 0 ~ 3.");
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
