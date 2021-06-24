// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernel_selector_utils.h"
#include "reorder/reorder_weights_kernel_selector.h"
#include "reorder/reorder_kernel_base.h"
#include "convolution/convolution_params.h"
#include <vector>
#include <memory>

namespace kernel_selector {

enum WeightsFormatSupportType { SUPPORTED, REORDER_NEEDED, UNSUPPORTED };

static WeightsType DataTypeToWeightsType(Datatype t) {
    switch (t) {
        case Datatype::UINT8:
            return WeightsType::UINT8;
        case Datatype::INT8:
            return WeightsType::INT8;
        case Datatype::F16:
            return WeightsType::F16;
        case Datatype::F32:
            return WeightsType::F32;
        case Datatype::BINARY:
            return WeightsType::BINARY;
        default:
            return WeightsType::UNSUPPORTED;
    }
}

static WeightsFormatSupportType CheckWeights(const weight_bias_params& newParams,
                                             WeightsType reqType,
                                             WeightsLayout reqLayouts,
                                             const ParamsKey& paramsKey,
                                             bool rotate) {
    // validate if weights type is image and if device supports requested sizes
    if (Tensor::IsImageType(reqLayouts)) {
        if (!CheckImageSize(newParams, reqLayouts))
            return UNSUPPORTED;
    }

    const auto& tensor = newParams.weights;
    const auto pitchesDifferFromLS = tensor.PitchesDifferFromLogicalDims();
    bool reorderNeeded = false;

    if ((reqType != tensor.GetDType()) && !(paramsKey.isEnabledDifferentInputWeightsTypes())) {
        reorderNeeded |= true;
    }

    reorderNeeded |= tensor.GetLayout() != reqLayouts;
    reorderNeeded |= rotate;

    if (reorderNeeded && !pitchesDifferFromLS && !rotate) {
        reorderNeeded = !((reqLayouts == WeightsLayout::io && tensor.GetLayout() == WeightsLayout::iyxo) ||
                          (reqLayouts == WeightsLayout::oi && tensor.GetLayout() == WeightsLayout::oiyx));
    }

    return reorderNeeded ? REORDER_NEEDED : SUPPORTED;
}

std::vector<size_t> GetImageSizes(const kernel_selector::WeightsTensor& dimensions, const WeightsLayout layout) {
    auto ofm = dimensions.OFM().v;
    auto ifm = dimensions.IFM().v;
    auto x = dimensions.X().v;
    auto y = dimensions.Y().v;

    switch (layout) {
        case WeightsLayout::image_2d_weights_c1_b_fyx:
        case WeightsLayout::image_2d_weights_c4_fyx_b:
            return {ofm, ifm * x * y};
        case WeightsLayout::image_2d_weights_winograd_6x3_s1_fbxyb:
            return {ofm * x * y * 8 / 3, ifm};
        case WeightsLayout::image_2d_weights_winograd_6x3_s1_xfbyb:
            return {ofm * y, ifm * x * 8 / 3};
        default:
            return {0, 0};
    }
}

bool CheckImageSize(const weight_bias_params& newParams, const WeightsLayout layout) {
    if (!newParams.engineInfo.bImageSupport)
        return false;

    auto image_sizes = GetImageSizes(newParams.weights, layout);
    if (image_sizes[0] == 0 || image_sizes[1] == 0 || image_sizes[0] > newParams.engineInfo.maxImage2dWidth ||
        image_sizes[1] > newParams.engineInfo.maxImage2dHeight)
        return false;

    return true;
}

bool UpdateWeightsParams(weight_bias_params& newParams,
                         const optional_params& options,
                         WeightsLayout reqLayout,
                         WeightsReorderParams& weightsReorderParams,
                         const ParamsKey& paramsKey,
                         size_t groups,
                         bool rotate) {
    const auto& optParams = static_cast<const weight_bias_optional_params&>(options);
    const auto inType = DataTypeToWeightsType(newParams.inputs[0].GetDType());
    const auto dtype = paramsKey.isEnabledDifferentInputWeightsTypes() ? newParams.weights.GetDType() : inType;
    switch (CheckWeights(newParams, inType, reqLayout, paramsKey, rotate)) {
        case SUPPORTED:
            return true;
        case UNSUPPORTED:
            return false;
        case REORDER_NEEDED: {
            if (!optParams.allowStaticInputReordering) {
                return false;
            }

            auto& reorderKS = ReorderWeightsKernelSelctor::Instance();
            reorder_weights_params r_params;

            r_params.layerID = newParams.layerID + "_reorder_";
            r_params.input = newParams.weights;
            r_params.output = newParams.weights.TransformIgnorePadding(reqLayout, dtype, groups, false);
            r_params.rotate_180 = rotate;
            r_params.engineInfo = newParams.engineInfo;

            reorder_optional_params op;
            KernelsData kernels_data = reorderKS.GetBestKernels(r_params, op);

            if (kernels_data.empty()) {
                throw std::runtime_error("No suitable kernel found for weights reorder from " +
                                         toString(r_params.input.GetLayout()) + " to " +
                                         toString(r_params.output.GetLayout()) +
                                         (rotate ? " with rotate" : ""));
            }

            weightsReorderParams.engine = WeightsReorderParams::Engine::GPU;
            weightsReorderParams.clKernel = std::make_shared<clKernelData>(kernels_data[0].kernels[0]);
            weightsReorderParams.dest = r_params.output;

            newParams.weights = newParams.weights.TransformIgnorePadding(reqLayout, dtype, groups);
            return true;
        }
    }

    return false;
}

JitConstants GetTensorFriendlyWorkGroupsJit(const DataTensor& t) {
    auto b = DataTensor::Channelndex(t.GetLayout(), Tensor::DataChannelName::BATCH);
    auto f = DataTensor::Channelndex(t.GetLayout(), Tensor::DataChannelName::FEATURE);
    auto x = DataTensor::Channelndex(t.GetLayout(), Tensor::DataChannelName::X);

    int gws_batch = -1;
    int gws_feature = -1;
    int gws_spatial = -1;

    int idx = 0;
    for (size_t i = 0; i < t.GetDims().size(); i++) {
        if (b == static_cast<int>(i))
            gws_batch = idx++;
        if (f == static_cast<int>(i))
            gws_feature = idx++;
        if (x == static_cast<int>(i))
            gws_spatial = idx++;
    }

    if (-1 == gws_batch)
        gws_batch = idx++;
    if (-1 == gws_feature)
        gws_feature = idx++;
    if (-1 == gws_spatial)
        gws_spatial = idx++;

    JitConstants jit{
        MakeJitConstant("GWS_BATCH", gws_batch),
        MakeJitConstant("GWS_FEATURE", gws_feature),
        MakeJitConstant("GWS_YX", gws_spatial),
    };

    return jit;
}

std::vector<size_t> GetTensorFriendlyWorkGroups(const DataTensor& t) {
    std::vector<size_t> sizes;
    auto y = DataTensor::Channelndex(t.GetLayout(), Tensor::DataChannelName::Y);
    auto z = DataTensor::Channelndex(t.GetLayout(), Tensor::DataChannelName::Z);
    auto w = DataTensor::Channelndex(t.GetLayout(), Tensor::DataChannelName::W);
    for (size_t i = 0; i < t.GetDims().size(); i++) {
        const auto& o = t.GetDims()[i];
        if (y == static_cast<int>(i) || z == static_cast<int>(i) || w == static_cast<int>(i)) {
            sizes.back() *= o.v;
        } else {
            sizes.push_back(o.v);
        }
    }

    for (size_t i = sizes.size(); i < 3; i++) {
        sizes.push_back(1U);
    }

    return sizes;
}

std::vector<size_t> GetOptimalLocalWorkGroupSizes(std::vector<size_t> gws, const EngineInfo& info) {
    const size_t lws_max = info.maxWorkGroupSize;
    const size_t optimal_lws_values[] = {256, 227, 224, 192, 160, 128, 96, 64, 32, 16, 8, 7, 6, 5, 4, 2, 1};
    size_t total_lws = 1;
    std::vector<size_t> lws;
    for (size_t i = 0; i < gws.size(); ++i) {
        auto rest_lws = lws_max / total_lws;
        size_t lws_idx = 0;
        while (rest_lws < optimal_lws_values[lws_idx]) lws_idx++;

        while (gws[i] % optimal_lws_values[lws_idx]) lws_idx++;

        lws.push_back(optimal_lws_values[lws_idx]);
        total_lws *= optimal_lws_values[lws_idx];
    }

    return lws;
}

bool CheckInputsOutputNoPitchSameDims(const base_params& params) {
    bool no_pitch_same_dims = true;

    std::map<DataLayout, std::pair<int, int>> block_layouts {
        {DataLayout::b_fs_yx_fsv16,          {1, 16}},
        {DataLayout::b_fs_zyx_fsv16,         {1, 16}},
        {DataLayout::b_fs_yx_fsv32,          {1, 32}},
        {DataLayout::b_fs_zyx_fsv32,         {1, 32}},
        {DataLayout::bs_fs_yx_bsv16_fsv16,   {16, 16}},
        {DataLayout::bs_fs_zyx_bsv16_fsv16,  {16, 16}},
        {DataLayout::bs_f_bsv8__af8,         {8, 8}},
        {DataLayout::bs_f_bsv16__af8,        {16, 8}},
        {DataLayout::b_fs_yx_fsv4,           {1, 4}},
        {DataLayout::fs_b_yx_fsv32,          {1, 32}},
        {DataLayout::b_fs_yx_32fp,           {1, 32}}
    };

    if (params.inputs.size()) {
        no_pitch_same_dims = !params.inputs[0].PitchesDifferFromLogicalDims();

        auto block_layout = block_layouts.find(params.inputs[0].GetLayout());
        if (block_layout != block_layouts.end()) {
            auto block_size = block_layout->second;
            if (params.inputs[0].Batch().v % block_size.first != 0 || params.inputs[0].Feature().v % block_size.second != 0)
                    return false;
        }

        if (params.fused_ops.size()) {
            for (auto fused_op : params.fused_ops) {
                for (size_t in = 0; in < fused_op.tensors.size(); in++) {
                    if (fused_op.tensors[in].LogicalSize() == 1)
                        continue;

                    auto layout = block_layouts.find(fused_op.tensors[in].GetLayout());
                    if (layout != block_layouts.end()) {
                        auto block_size = layout->second;
                        if (fused_op.tensors[in].Batch().v % block_size.first != 0 || fused_op.tensors[in].Feature().v % block_size.second != 0)
                            return false;
                    }

                    no_pitch_same_dims = no_pitch_same_dims && (params.inputs[0] == fused_op.tensors[in]);
                }
            }
        }

        for (size_t i = 1; i < params.inputs.size(); i++) {
            no_pitch_same_dims = no_pitch_same_dims && (params.inputs[0] == params.inputs[i]);

            auto layout = block_layouts.find(params.inputs[i].GetLayout());
            if (layout != block_layouts.end()) {
                auto block_size = layout->second;
                if (params.inputs[i].Batch().v % block_size.first != 0 || params.inputs[i].Feature().v % block_size.second != 0)
                    return false;
            }
        }

        no_pitch_same_dims = no_pitch_same_dims && (params.inputs[0] == params.output);
    }

    return no_pitch_same_dims;
}
}  // namespace kernel_selector
