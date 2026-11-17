// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rms_kernel_bfyx_opt.h"
#include "kernel_selector_utils.h"
#include <string>

namespace {
    std::vector<kernel_selector::fused_operation_desc>::const_iterator get_dq_it(const kernel_selector::rms_params& params) {
        return std::find_if(params.fused_ops.begin(), params.fused_ops.end(), [](const kernel_selector::fused_operation_desc& f) {
            return f.GetType() == kernel_selector::KernelType::DYNAMIC_QUANTIZE;
        });
    }

    bool has_dynamic_quantize_post_op(const kernel_selector::rms_params& params) {
        return get_dq_it(params) != params.fused_ops.end();
    }
} // namespace

namespace kernel_selector {
static constexpr size_t subgroup_size = 16;

static bool is_feature_axis_normalization(const rms_params& params) {
    const auto axis = params.axis < 0 ? params.axis + params.ov_input_rank : params.axis;
    return axis == 1 && (params.ov_input_rank == 4 || params.ov_input_rank == 5);
}

// Compute maximum possible LWS that does not exceed device capabilities and optimizes number of global memory reads
static std::pair<size_t, size_t> get_item_num_and_lws(const rms_params params, size_t data_size) {
    size_t lws = 1;
    size_t itemsNum = data_size;
    const auto& input = params.inputs[0];
    auto local_mem_per_wi = 2 * BytesPerElement(input.GetDType());
    auto max_lws = std::min(params.engineInfo.maxWorkGroupSize, params.engineInfo.maxLocalMemSize / local_mem_per_wi);

    while ((itemsNum > 8 || lws < itemsNum) && (2 * lws <= max_lws)) {
        if (has_dynamic_quantize_post_op(params) && (itemsNum / 2) % 2 == 1) {
            break;
        }
        lws *= 2;
        itemsNum /= 2;
    }
    return {itemsNum, lws};
}

// On the feature axis of an fsv16-blocked tensor (b_fs_yx_fsv16 / b_fs_zyx_fsv16) with no feature padding,
// each fsv16 block stores 16 consecutive features at stride 1, and XBLK adjacent X elements sit at stride 16,
// so one SIMD16 sub-group can block-read/write XBLK spatial columns at once.
static bool is_fsv16_feature_axis_blockable(const rms_params& params) {
    if (params.has_dynamic_tensors() || !is_feature_axis_normalization(params)) {
        return false;
    }
    const auto& in = params.inputs[0];
    const auto& out = params.outputs[0];
    const auto in_l = in.GetLayout();
    if ((in_l != DataLayout::b_fs_yx_fsv16 && in_l != DataLayout::b_fs_zyx_fsv16) || out.GetLayout() != in_l) {
        return false;
    }
    if (in.Feature().v == 0 || in.Feature().v % subgroup_size != 0 || in.Feature().v != out.Feature().v) {
        return false;
    }
    if (in.Feature().pad.before != 0 || in.Feature().pad.after != 0 || out.Feature().pad.before != 0 || out.Feature().pad.after != 0) {
        return false;
    }
    if (params.elementwise_affine) {
        const auto& gamma = params.inputs[1];
        const auto n = gamma.LogicalSize();
        if (n != 1 && n != in.Feature().v) {
            return false;
        }
    }
    return !has_dynamic_quantize_post_op(params);
}

struct Fsv16Tiling {
    size_t xblk = 1;
    size_t fsplit = 1;
    size_t blocks_per_sg = 1;
    bool cache = true;
};

static Fsv16Tiling get_fsv16_tiling(const rms_params& params) {
    constexpr size_t max_cached_per_lane = 64;
    constexpr size_t max_fsplit = 16;
    const auto& in = params.inputs[0];
    Fsv16Tiling t;
    const size_t x = in.X().v;
    // XBLK is the largest of 8/4/2/1 dividing X, so a tile never crosses a row end. Spatial padding (X
    // included) needs no check: in b_fs_(z)yx_fsv16 the X pitch is always 16 elements, padding only moves
    // where a row starts (applied by INPUT0_GET_INDEX/OUTPUT_GET_INDEX), so the XBLK columns of one fsv16
    // block remain XBLK * 16 contiguous elements that a single BLOCK_READN/BLOCK_WRITEN covers.
    for (size_t cand : {8, 4, 2, 1}) {
        if (x % cand == 0) {
            t.xblk = cand;
            break;
        }
    }
    const size_t fblocks = in.Feature().v / subgroup_size;
    const size_t tiles = in.Batch().v * in.Z().v * in.Y().v * (x / t.xblk);
    const size_t target_sgs = std::max<size_t>(params.engineInfo.computeUnitsCount, 1) * 32;
    const size_t max_split = std::min({max_fsplit, fblocks, params.engineInfo.maxWorkGroupSize / subgroup_size});

    auto per_sg = [&](size_t split) {
        return (fblocks + split - 1) / split;
    };
    size_t split = 1;
    while (split * 2 <= max_split && per_sg(split) * t.xblk > max_cached_per_lane) {
        split *= 2;
    }
    while (split * 2 <= max_split && tiles * split < target_sgs) {
        split *= 2;
    }
    t.fsplit = split;
    t.blocks_per_sg = per_sg(split);
    t.cache = t.blocks_per_sg * t.xblk <= max_cached_per_lane;
    return t;
}

ParamsKey RMSKernelBfyxOpt::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::BF16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::BF16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F8E4M3);
    k.EnableOutputDataType(Datatype::F8E5M2);
    k.EnableOutputDataType(Datatype::F8E8M0);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::bfzyx);
    // The feature-axis path addresses input and output through get_input_index()/get_output_index(), which
    // are layout-aware, so blocked tensors are served by the same code. The other path reads a normalized
    // row as contiguous memory and is therefore restricted to the plain layouts in Validate().
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableInputLayout(DataLayout::b_fs_zyx_fsv16);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfzyx);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_zyx_fsv16);
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

    bool gamma_is_scalar = false;
    if (params.elementwise_affine && !params.inputs[1].is_dynamic()) {
        gamma_is_scalar = params.inputs[1].LogicalSize() == 1;
    }
    jit.AddConstant(MakeJitConstant("RMS_GAMMA_IS_SCALAR", gamma_is_scalar));

    const bool feature_axis = is_feature_axis_normalization(params);
    const bool fsv16_fast = is_fsv16_feature_axis_blockable(params);
    jit.AddConstant(MakeJitConstant("RMS_FEATURE_AXIS", feature_axis));
    jit.AddConstant(MakeJitConstant("RMS_FSV16_FAST", fsv16_fast));
    if (fsv16_fast) {
        const auto t = get_fsv16_tiling(params);
        const auto& in = params.inputs[0];
        jit.AddConstants({
            MakeJitConstant("FSV", subgroup_size),
            MakeJitConstant("XBLK", t.xblk),
            MakeJitConstant("FSPLIT", t.fsplit),
            MakeJitConstant("BLOCKS_PER_SG", t.blocks_per_sg),
            MakeJitConstant("F_BLOCKS", in.Feature().v / subgroup_size),
            MakeJitConstant("X_BLOCKS", in.X().v / t.xblk),
            MakeJitConstant("CACHE_INPUT", t.cache),
            MakeJitConstant("INV_DATA_SIZE", 1.0F / static_cast<float>(in.Feature().v)),
        });
    }
    if (feature_axis) {
        if (!params.fused_ops.empty()) {
            std::vector<std::string> idx_order;
            if (fsv16_fast) {
                if (params.inputs[0].GetDims().size() == 5) {
                    idx_order = {"(b_idx)", "(f_base + lane)", "(z_idx)", "(y_idx)", "(x0 + xi)"};
                } else {
                    idx_order = {"(b_idx)", "(f_base + lane)", "(y_idx)", "(x0 + xi)"};
                }
            } else if (params.inputs[0].GetDims().size() == 5) {
                idx_order = {"(b_idx)", "(f_idx)", "(z_idx)", "(y_idx)", "(x_idx)"};
            } else {
                idx_order = {"(b_idx)", "(f_idx)", "(y_idx)", "(x_idx)"};
            }
            auto conf = FusedOpsConfiguration("", idx_order, "normalized", params.outputs[0].GetDType(), 1);
            jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
        }
    }

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

    if (has_padding) {
        jit.AddConstant(MakeJitConstant("HAS_PADDING", 1));
    }

    if (params.has_dynamic_tensors()) {
        const auto& input = params.inputs[0];
        DimensionAccessHelperJit dims(input);
        std::string data_size;
        if (feature_axis) {
            data_size = dims.f();
        } else {
            switch (params.ov_input_rank) {
            case 1:
                data_size = dims.b();
                break;
            case 2:
                data_size = dims.f();
                break;
            case 3:
                data_size = dims.y();
                break;
            default:
                data_size = dims.x();
                break;
            }
        }

        const std::string lws_0 = "get_local_size(0)";
        // data_size string starts digit when it has static dim.
        bool is_static_data_size = std::isdigit(data_size[0]) != 0;
        size_t stack_size = 33;
        if (is_static_data_size) {
            auto item_num_and_lws = get_item_num_and_lws(params, stoi(data_size));
            stack_size = cldnn::ceil_div(std::stoi(data_size), item_num_and_lws.second);
        }
        jit.AddConstants({
            MakeJitConstant("DATA_SIZE", data_size),
            MakeJitConstant("LWS", lws_0),
            MakeJitConstant("SLM_SIZE", dispatchData.maxSlmSize),
            MakeJitConstant("STACK_SIZE", stack_size)
        });
    } else {
        jit.AddConstants({
            MakeJitConstant("DATA_SIZE", dispatchData.dataSize),
            MakeJitConstant("LWS", dispatchData.lws[0]),
            MakeJitConstant("SLM_SIZE", dispatchData.lws[0]),
            MakeJitConstant("STACK_SIZE", dispatchData.itemsNum + 1)
        });
    }
    jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", subgroup_size));
    jit.AddConstant(MakeJitConstant("SUBGROUP_BLOCK_SIZE", dispatchData.subgroupBlockSize));
    if (!params.fused_ops.empty() && !feature_axis) {
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

        const bool has_dynamic_quantize = has_dynamic_quantize_post_op(params);
        const auto fused_output_type = has_dynamic_quantize ? Datatype::F32 : params.outputs[0].GetDType();
        auto conf = FusedOpsConfiguration("", idx_order, "normalized", fused_output_type, 1);
        jit.Merge(MakeFusedOpsJitConstants(params, { conf }));
        if (has_dynamic_quantize) {
            jit.AddConstant(MakeJitConstant("HAS_DYNAMIC_QUANTIZE", "1"));
            jit.AddConstant(MakeJitConstant("OUTPUT1", get_dq_it(params)->output_tensors[1]));
        }
    }

    return jit;
}

RMSKernelBase::DispatchData RMSKernelBfyxOpt::SetDefault(const rms_params& params) const {
    DispatchData dispatchData;
    const auto& input = params.inputs[0];

    auto local_mem_per_wi = 2 * BytesPerElement(input.GetDType());
    auto max_lws = std::min(params.engineInfo.maxWorkGroupSize, params.engineInfo.maxLocalMemSize / local_mem_per_wi);
    dispatchData.maxSlmSize = max_lws;
    if (!params.has_dynamic_tensors()) {
        if (is_fsv16_feature_axis_blockable(params)) {
            const auto t = get_fsv16_tiling(params);
            dispatchData.dataSize = input.Feature().v;
            dispatchData.dataCount = (input.X().v / t.xblk) * input.Y().v * input.Z().v * input.Batch().v;
            dispatchData.lws = {1, t.fsplit * subgroup_size, 1};
            dispatchData.gws = {dispatchData.dataCount, t.fsplit * subgroup_size, 1};
            dispatchData.itemsNum = t.blocks_per_sg;
            dispatchData.leftovers = 0;
            dispatchData.subgroupBlockSize = 1;
            return dispatchData;
        }
        // data size to be processed within a LWG
        if (is_feature_axis_normalization(params)) {
            dispatchData.dataSize = input.Feature().v;
            dispatchData.dataCount = input.Batch().v * input.Z().v * input.Y().v * input.X().v;
        } else {
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
        }
        dispatchData.gws[0] = 1;
        dispatchData.gws[1] = dispatchData.dataCount;
        dispatchData.gws[2] = 1;

        dispatchData.lws[0] = 1;
        dispatchData.lws[1] = 1;
        dispatchData.lws[2] = 1;

        auto item_num_and_lws = get_item_num_and_lws(params, dispatchData.dataSize);
        dispatchData.itemsNum = item_num_and_lws.first;
        dispatchData.lws[0] = item_num_and_lws.second;
        dispatchData.gws[0] = dispatchData.lws[0];
        dispatchData.leftovers = dispatchData.dataSize % dispatchData.lws[0];

        if (dispatchData.itemsNum >> 3) {
            dispatchData.subgroupBlockSize = 8;
        } else if (dispatchData.itemsNum >> 2) {
            dispatchData.subgroupBlockSize = 4;
        } else if (dispatchData.itemsNum >> 1) {
            dispatchData.subgroupBlockSize = 2;
        } else {
            dispatchData.subgroupBlockSize = 1;
        }
    } else {
        dispatchData.subgroupBlockSize = 8;
    }
    if (has_dynamic_quantize_post_op(params)) {
        OPENVINO_ASSERT(dispatchData.subgroupBlockSize != 1 && dispatchData.leftovers == 0 && subgroup_size == 16);
    }
    return dispatchData;
}

bool RMSKernelBfyxOpt::Validate(const Params& p) const {
    if (!Parent::Validate(p)) {
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    }

    const rms_params& params = static_cast<const rms_params&>(p);
    const bool feature_axis = is_feature_axis_normalization(params);
    const auto in_layout = params.inputs[0].GetLayout();
    if (in_layout != DataLayout::bfyx && in_layout != DataLayout::bfzyx && !feature_axis) {
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    }
    if (params.elementwise_affine) {
        const auto& gamma = params.inputs[1];

        if (!gamma.is_dynamic()) {
            size_t data_size = gamma.LogicalSize();
            // The feature axis reads gamma scalar per lane; only the row-major path block-reads it.
            if (!feature_axis && data_size != 1 && data_size < subgroup_size) {
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
