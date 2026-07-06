// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rms_kernel_bfyx_opt.h"
#include "kernel_selector_utils.h"
#include <algorithm>
#include <cctype>
#include <string>

namespace kernel_selector {
static constexpr size_t subgroup_size = 16;
static constexpr size_t target_items_per_wi = 8;
static constexpr size_t max_register_stack = 16;

// Generalized aboutSHW rule: largest power-of-two LWS that keeps at least 8 normalized
// elements per work-item. RMS is subgroup-based, so clamp the minimum LWS to one subgroup.
static size_t get_generalized_lws(const rms_params& params, size_t data_size) {
    size_t lws = subgroup_size;
    const auto& input = params.inputs[0];
    auto local_mem_per_wi = 2 * BytesPerElement(input.GetDType());
    auto max_lws = std::min(params.engineInfo.maxWorkGroupSize, params.engineInfo.maxLocalMemSize / local_mem_per_wi);

    const size_t limit = std::max(subgroup_size, std::min(max_lws, data_size / target_items_per_wi));
    while (2 * lws <= limit) {
        lws *= 2;
    }
    return lws;
}

static size_t get_stack_size(size_t data_size, size_t lws) {
    return (data_size + lws - 1) / lws;
}

static bool is_decimal_number(const std::string& value) {
    return !value.empty() && std::all_of(value.begin(), value.end(), [](char c) {
        return std::isdigit(static_cast<unsigned char>(c));
    });
}

ParamsKey RMSKernelBfyxOpt::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::bfzyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfzyx);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDifferentTypes();
    k.EnableDynamicShapesSupport();
    return k;
}

DeviceFeaturesKey RMSKernelBfyxOpt::get_required_device_features_key(const Params& params) const {
    DeviceFeaturesKey k;
    k.requires_subgroups();
    k.requires_subgroup_reduce();
    k.requires_reqd_subgroup_size();

    return k;
}

JitConstants RMSKernelBfyxOpt::GetJitConstants(const rms_params& params, DispatchData dispatchData) const {
    auto jit = Parent::GetJitConstants(params, dispatchData);

    // Check for any padding (dynamic or static) on input dimensions.
    // The flat addressing path (data_idx * data_size) assumes contiguous memory,
    // which breaks when padding introduces gaps between slices (e.g., from in-place crop).
    // Switch to per-dimension indexed addressing via get_input_index() in that case.
    bool has_padding = false;
    for (const auto& dim : params.inputs[0].GetDims()) {
        if (dim.pad.is_dynamic || dim.pad.before != 0 || dim.pad.after != 0) {
            has_padding = true;
            break;
        }
    }

    if (has_padding)
        jit.AddConstant(MakeJitConstant("HAS_PADDING", 1));

    if (params.has_dynamic_tensors()) {
        const auto& input = params.inputs[0];
        DimensionAccessHelperJit dims(input);
        std::string data_size;
        switch (params.ov_input_rank) {
            case 1 :
                data_size = dims.b();
                break;
            case 2 :
                data_size = dims.f();
                break;
            case 3 :
                data_size = dims.y();
                break;
            default:
                data_size = dims.x();
                break;
        }

        std::string lws_0 = "get_local_size(0)";
        // data_size string starts digit when it has static dim.
        bool is_static_data_size = is_decimal_number(data_size);
        size_t stack_size = max_register_stack;
        bool reread_input = true;
        bool one_subgroup_row = false;
        bool multi_subgroup_row = false;
        if (is_static_data_size) {
            const size_t static_data_size = std::stoul(data_size);
            const size_t lws = get_generalized_lws(params, static_data_size);
            const size_t required_stack = get_stack_size(static_data_size, lws);
            lws_0 = std::to_string(lws);
            stack_size = std::min(required_stack, max_register_stack);
            reread_input = required_stack > max_register_stack;
            one_subgroup_row = lws == subgroup_size;
            multi_subgroup_row = !one_subgroup_row;
        }
        jit.AddConstants({
            MakeJitConstant("DATA_SIZE", data_size),
            MakeJitConstant("LWS", lws_0),
            MakeJitConstant("SLM_SIZE", dispatchData.maxSlmSize),
            MakeJitConstant("STACK_SIZE", stack_size),
            MakeJitConstant("SUBGROUP_BLOCK_SIZE", 8),
            MakeJitConstant("ONE_SUBGROUP_ROW", one_subgroup_row),
            MakeJitConstant("MULTI_SUBGROUP_ROW", multi_subgroup_row),
            MakeJitConstant("RMS_REREAD_INPUT", reread_input),
        });
    } else {
        const size_t stack_size = get_stack_size(dispatchData.dataSize, dispatchData.lws[0]);
        jit.AddConstants({
            MakeJitConstant("DATA_SIZE", dispatchData.dataSize),
            MakeJitConstant("LWS", dispatchData.lws[0]),
            MakeJitConstant("SLM_SIZE", dispatchData.maxSlmSize),
            MakeJitConstant("STACK_SIZE", std::min(stack_size, max_register_stack)),
            MakeJitConstant("SUBGROUP_BLOCK_SIZE", 8),
            MakeJitConstant("ONE_SUBGROUP_ROW", dispatchData.lws[0] == subgroup_size),
            MakeJitConstant("MULTI_SUBGROUP_ROW", dispatchData.lws[0] != subgroup_size),
            MakeJitConstant("RMS_REREAD_INPUT", stack_size > max_register_stack),
        });
    }
    jit.AddConstant(MakeJitConstant("INPUT_RANK", params.ov_input_rank));
    jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", subgroup_size));
    if (!params.fused_ops.empty()) {
        switch (params.ov_input_rank) {
            case 1 :
                jit.AddConstant(MakeJitConstant("LAST_DIM", "b"));
                break;
            case 2 :
                jit.AddConstant(MakeJitConstant("LAST_DIM", "f"));
                break;
            case 3 :
                jit.AddConstant(MakeJitConstant("LAST_DIM", "y"));
                break;
            default:
                jit.AddConstant(MakeJitConstant("LAST_DIM", "x"));
                break;
        }

        std::vector<std::string> idx_order;
        if (params.inputs[0].GetDims().size() == 5) {
            idx_order = { "(b)", "(f)", "(z)", "(y)", "(x)" };
        } else if (params.inputs[0].GetDims().size() <= 4) {
            idx_order = { "(b)", "(f)", "(y)", "(x)" };
        } else {
            OPENVINO_THROW("rms_bfyx_opt doesn't support 5D or higher dims.");
        }

        auto conf = FusedOpsConfiguration("", idx_order, "normalized", params.outputs[0].GetDType(), 1);
        jit.Merge(MakeFusedOpsJitConstants(params, { conf }));
    }

    return jit;
}

RMSKernelBase::DispatchData RMSKernelBfyxOpt::SetDefault(const rms_params& params) const {
    DispatchData dispatchData;
    const auto& input = params.inputs[0];

    auto local_mem_per_wi = 2 * BytesPerElement(input.GetDType());
    auto max_lws = std::min(params.engineInfo.maxWorkGroupSize, params.engineInfo.maxLocalMemSize / local_mem_per_wi);
    dispatchData.maxSlmSize = max_lws;
    // data size to be processed within a LWG. For dynamic kernels, these values are
    // populated during dispatch update once concrete dimensions are known; if a dimension
    // is still unknown, leave the default dynamic dispatch data untouched.
    switch (params.ov_input_rank) {
        case 1:
            dispatchData.dataSize = input.Batch().v;
            dispatchData.dataCount = 1;
            break;
        case 2:
            dispatchData.dataSize = input.Feature().v;
            dispatchData.dataCount = input.Batch().v;
            break;
        case 3:
            dispatchData.dataSize = input.Y().v;
            dispatchData.dataCount = input.Batch().v * input.Feature().v;
            break;
        default:
            dispatchData.dataSize = input.X().v;
            dispatchData.dataCount = input.Batch().v * input.Feature().v * input.Z().v * input.Y().v;
            break;
    }

    if (dispatchData.dataSize != 0 && dispatchData.dataCount != 0) {
        dispatchData.gws[0] = 1;
        dispatchData.gws[1] = dispatchData.dataCount;
        dispatchData.gws[2] = 1;

        dispatchData.lws[0] = 1;
        dispatchData.lws[1] = 1;
        dispatchData.lws[2] = 1;

        dispatchData.lws[0] = get_generalized_lws(params, dispatchData.dataSize);
        dispatchData.itemsNum = dispatchData.dataSize / dispatchData.lws[0];
        dispatchData.gws[0] = dispatchData.lws[0];
        dispatchData.leftovers = dispatchData.dataSize % dispatchData.lws[0];
        dispatchData.subgroupBlockSize = 8;
    } else {
        dispatchData.subgroupBlockSize = 8;
    }
    return dispatchData;
}

bool RMSKernelBfyxOpt::Validate(const Params& p) const {
    if (!Parent::Validate(p))
        DO_NOT_USE_THIS_KERNEL(p.layerID);

    const rms_params& params = static_cast<const rms_params&>(p);
    if (params.elementwise_affine) {
        const auto& gamma = params.inputs[1];

        if (!gamma.is_dynamic()) {
            size_t data_size = gamma.LogicalSize();
            if (data_size < subgroup_size) {
                DO_NOT_USE_THIS_KERNEL(p.layerID);
            }
        }
    }
    return true;
}

KernelsData RMSKernelBfyxOpt::GetKernelsData(const Params& params) const {
    return GetCommonKernelsData(params);
}

KernelsPriority RMSKernelBfyxOpt::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_7;
}
}  // namespace kernel_selector
