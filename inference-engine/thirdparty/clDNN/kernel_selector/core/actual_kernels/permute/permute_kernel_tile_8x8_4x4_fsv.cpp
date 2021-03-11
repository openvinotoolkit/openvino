// Copyright (c) 2016-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#include "permute_kernel_tile_8x8_4x4_fsv.h"
#include "kernel_selector_utils.h"
#include <string>
#include <functional>
#include <cmath>
#include <iostream>

// Tile size : 4x4 or 8x8
#define MIN_TILE_SIZE 4
#define DEFAULT_TILE_SIZE 8
#define MAX_TILE_SIZE 16

#define CEIL_DIV(A, B) ((A + B - 1)/(B))

namespace kernel_selector {

ParamsKey PermuteKernel_tile_8x8_4x4_fsv::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableInputDataType(Datatype::INT32);
    k.EnableInputDataType(Datatype::INT64);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::INT32);
    k.EnableOutputDataType(Datatype::INT64);
    k.EnableDifferentTypes();
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableInputLayout(DataLayout::b_fs_zyx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_zyx_fsv16);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv32);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv32);
    k.EnableInputLayout(DataLayout::b_fs_zyx_fsv32);
    k.EnableOutputLayout(DataLayout::b_fs_zyx_fsv32);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv4);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv4);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    return k;
}

static inline std::vector<std::string> GetFusedOpOrderVector(size_t size) {
    std::vector<std::string> ret;
    switch (size) {
        case 4:
            ret = {"b", "y + lh", "x", "f + lw"};
            break;
        case 5:
            ret = {"b", "z + lh", "y", "x", "f + lw"};
            break;
        default : throw std::runtime_error("Unsupported combination\n");
    }
    return ret;
}

static inline std::string GetTiledOutputOrder(size_t size) {
    std::string order_str = "";
    switch (size) {
        case 4 :
            order_str = "b, y, x, f + lw";
            break;
        case 5 :
            order_str = "b, z, y, x, f + lw";
            break;
        default : throw std::runtime_error("Unsupported combination\n");
    }
    return order_str;
}

static inline std::string GetTiledInputOrder(size_t size) {
    std::string order_str = "";
    switch (size) {
        case 4 :
            order_str = "b, f, y + lh, x";
            break;
        case 5 :
            order_str = "b, f, z + lh, y, x";
            break;
        default : throw std::runtime_error("Unsupported combination\n");
    }
    return order_str;
}

static inline size_t GetFsvAlignment(const permute_params& params) {
    const auto& in =  params.inputs[0];
    int fsv_alignment = -1;
    switch (in.GetLayout()) {
        case DataLayout::b_fs_yx_fsv16:
        case DataLayout::b_fs_zyx_fsv16:
            fsv_alignment = 16;
            break;
        case DataLayout::b_fs_yx_fsv32:
        case DataLayout::b_fs_zyx_fsv32:
            fsv_alignment = 32;
            break;
        case DataLayout::b_fs_yx_fsv4:
            fsv_alignment = 32;
            break;
        default:
            throw std::runtime_error("Unsupported combination\n");
    }
    return fsv_alignment;
}

static inline size_t GetTileSize(const permute_params& params) {
    // vector type support at most 16 elements
    // u8 and i8 use the largest vector type: min(16. fsv_alignment)
    if (((params.inputs[0].GetDType() == Datatype::UINT8) && (params.output.GetDType() == Datatype::UINT8)) ||
        ((params.inputs[0].GetDType() == Datatype::INT8) && (params.output.GetDType() == Datatype::INT8)) ||
        ((params.inputs[0].GetDType() == Datatype::F16) && (params.output.GetDType() == Datatype::F16))) {
        return std::min(static_cast<size_t>(MAX_TILE_SIZE), GetFsvAlignment(params));
    }

    if ((params.inputs[0].GetDType() == Datatype::INT64) || (params.output.GetDType() == Datatype::INT64)) {
        return MIN_TILE_SIZE;
    }

    // supports 4x4 or 8x8 tiling
    size_t rotating_dim = 0;
    if (params.inputs[0].GetDims().size() == 4) {
        rotating_dim = params.inputs[0].Y().v;
    } else if (params.inputs[0].GetDims().size() == 5) {
        rotating_dim = params.inputs[0].Z().v;
    }

    if (rotating_dim < DEFAULT_TILE_SIZE && params.inputs[0].Feature().v < DEFAULT_TILE_SIZE) {
        return MIN_TILE_SIZE;
    }

    return DEFAULT_TILE_SIZE;
}

JitConstants PermuteKernel_tile_8x8_4x4_fsv::GetJitConstants(const permute_params& params, const CommonDispatchData& dispatchData) const {
    auto jit = Parent::GetJitConstants(params, dispatchData);
    const size_t f = params.inputs[0].Feature().v;
    const size_t z = params.inputs[0].Z().v;
    const size_t y = params.inputs[0].Y().v;
    const size_t x = params.inputs[0].X().v;

    const size_t tile_width = GetTileSize(params);
    const size_t tile_height = tile_width;

    // Note: this is default mode and different vector width is not being used now.
    const uint64_t total_lws = dispatchData.lws[0] * dispatchData.lws[1] * dispatchData.lws[2];
    const uint64_t transpose_buffer_size = tile_width*tile_height * total_lws;

    const size_t input_ndims = params.inputs[0].GetDims().size();
    const size_t output_ndims = params.output.GetDims().size();

    const size_t fsv_alignment = GetFsvAlignment(params);

    jit.AddConstant(MakeJitConstant("INPUT0_TILED_ORDER", GetTiledInputOrder(input_ndims)));
    jit.AddConstant(MakeJitConstant("OUTPUT_TILED_ORDER", GetTiledOutputOrder(output_ndims)));

    jit.AddConstant(MakeJitConstant("INPUT0_SIZE_FS", CEIL_DIV(f, fsv_alignment)));
    jit.AddConstant(MakeJitConstant("INPUT0_SIZE_F", f));
    jit.AddConstant(MakeJitConstant("TILE_WIDTH", tile_width));
    jit.AddConstant(MakeJitConstant("TILE_HEIGHT", tile_height));
    jit.AddConstant(MakeJitConstant("TILE_STRIDE", fsv_alignment * x));
    jit.AddConstant(MakeJitConstant("FSV_ALIGNMENT", fsv_alignment));
    jit.AddConstant(MakeJitConstant("GROUP_STRIDE", tile_width * x * fsv_alignment));
    jit.AddConstant(MakeJitConstant("INPUTVTYPE", "CAT(INPUT0_TYPE, TILE_WIDTH)"));
    jit.AddConstant(MakeJitConstant("OUTPUTVTYPE", "CAT(OUTPUT_TYPE, TILE_HEIGHT)"));
    jit.AddConstant(MakeJitConstant("VLOAD", "CAT(vload, TILE_WIDTH)"));
    jit.AddConstant(MakeJitConstant("VSTORE", "CAT(vstore, TILE_HEIGHT)"));
    jit.AddConstant(MakeJitConstant("AS_INPUTVTYPE", "CAT(as_, INPUTVTYPE)"));
    jit.AddConstant(MakeJitConstant("AS_OUTPUTVTYPE", "CAT(as_, OUTPUTVTYPE)"));
    jit.AddConstant(MakeJitConstant("LOCAL_BUF_STRIDE", tile_width*tile_height));
    jit.AddConstant(MakeJitConstant("TRANS_BUF_SIZE", transpose_buffer_size / tile_height));

    // whether F is tile_width-aligned
    if (f % tile_width != 0) {
        jit.AddConstant(MakeJitConstant("F_REMAINDER_SIZE", f % tile_width));
        jit.AddConstant(MakeJitConstant("F_REMAINDER_CONDITION", "((INPUT0_SIZE_F - F_REMAINDER_SIZE) <= f) && (f < INPUT0_SIZE_F)"));
        jit.AddConstant(MakeJitConstant("F_NO_REMAINDER_CONDITION", "(f < (INPUT0_SIZE_F - F_REMAINDER_SIZE))"));
    } else {
        jit.AddConstant(MakeJitConstant("F_NO_REMAINDER_CONDITION", "(f < INPUT0_SIZE_F)"));
    }

    // whether y (or z if b_fs_zyx_fsv16) is tile_height-aligned
    if ((input_ndims == 4) && (y % tile_height != 0)) {
        jit.AddConstant(MakeJitConstant("YZ_REMAINDER_SIZE", y % tile_height));
        jit.AddConstant(MakeJitConstant("YZ_NO_REMAINDER_CONDITION", "y < (INPUT0_SIZE_Y - YZ_REMAINDER_SIZE)"));
        jit.AddConstant(MakeJitConstant("YZ_REMAINDER_CONDITION", "((INPUT0_SIZE_Y - YZ_REMAINDER_SIZE) <= y) && (y < INPUT0_SIZE_Y)"));
    } else if ((input_ndims == 5) && (z % tile_height != 0)) {
        jit.AddConstant(MakeJitConstant("YZ_REMAINDER_SIZE", z % tile_height));
        jit.AddConstant(MakeJitConstant("YZ_NO_REMAINDER_CONDITION", "z < (INPUT0_SIZE_Z - YZ_REMAINDER_SIZE)"));
        jit.AddConstant(MakeJitConstant("YZ_REMAINDER_CONDITION", "((INPUT0_SIZE_Z - YZ_REMAINDER_SIZE) <= z) && (z < INPUT0_SIZE_Z)"));
    }

    if (!params.fused_ops.empty()) {
        std::vector<std::string> output_order = GetFusedOpOrderVector(output_ndims);
        FusedOpsConfiguration conf = {"", output_order, "input_var", params.inputs[0].GetDType(), 1};
        jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
    }
    return jit;
}

static std::vector<size_t> GetBestLwsFromGws(const permute_params& params, const std::vector<size_t>& gws, const size_t tile_width, const size_t tile_height) {
    std::vector<size_t> lws{1, 1, 1};
    std::vector<size_t> dims{0, 1, 2};

    // SLM size: elemsize * tile_width * tile_width * work_items <= 64K
    const size_t elem_size = sizeof(params.output.GetDType());
    const size_t max_local_mem_size = params.engineInfo.maxLocalMemSize;
    const size_t max_work_group_size = params.engineInfo.maxWorkGroupSize;
    size_t max_num_work_items = std::min(max_work_group_size, max_local_mem_size / (elem_size * tile_width * tile_height));

    for (size_t i = 0; i < dims.size(); ++i) {
        size_t dim = dims[i];
        size_t max_divider = static_cast<size_t>(std::sqrt(gws[dim]) + 1);
        for (size_t divider = 1; divider <= max_divider; ++divider) {
            if (gws[dim] % divider == 0) {
                const size_t lws0 = gws[dim] / divider;
                if (lws0 <= max_num_work_items) {
                    lws[dim] = std::max(lws[dim], lws0);
                }
                if (divider <= max_num_work_items) {
                    lws[dim] = std::max(lws[dim], divider);
                }
            }
        }
        max_num_work_items /= lws[dim];
    }
    return lws;
}

static inline std::vector<size_t> GetGWS(const permute_params& params) {
    const auto& in =  params.inputs[0];
    const size_t tile_width = GetTileSize(params);
    const size_t tile_height = tile_width;
    const size_t fsv_alignment = GetFsvAlignment(params);
    std::vector<size_t> gws;
    switch (in.GetLayout()) {
        case DataLayout::b_fs_yx_fsv16:
        case DataLayout::b_fs_yx_fsv32:
        case DataLayout::b_fs_yx_fsv4:
            gws = {CEIL_DIV(fsv_alignment, tile_width),
                CEIL_DIV(in.Y().v, tile_height) * in.X().v,
                in.Batch().v * CEIL_DIV(in.Feature().v, fsv_alignment)};
            break;
        case DataLayout::b_fs_zyx_fsv16:
        case DataLayout::b_fs_zyx_fsv32:
            gws = {CEIL_DIV(fsv_alignment, tile_width),
                CEIL_DIV(in.Z().v, tile_height) * in.X().v * in.Y().v,
                in.Batch().v * CEIL_DIV(in.Feature().v, fsv_alignment)};
            break;
        default:
            throw std::runtime_error("Unsupported combination\n");
    }
    return gws;
}

CommonDispatchData PermuteKernel_tile_8x8_4x4_fsv::SetDefault(const permute_params& params) const {
    CommonDispatchData dispatchData;
    const size_t tile_width = GetTileSize(params);
    const size_t tile_height = tile_width;
    dispatchData.gws = GetGWS(params);
    dispatchData.lws = GetBestLwsFromGws(params, dispatchData.gws, tile_width, tile_height);
    return dispatchData;
}

// Validate is the same as permute_kernel_tile_8x8_4x4
bool PermuteKernel_tile_8x8_4x4_fsv::Validate(const Params& p, const optional_params& o) const {
    if (!Parent::Validate(p, o)) return false;

    std::function<bool(const std::vector<uint16_t>&)> is_rotating_except_batch = [](const std::vector<uint16_t>& order) {
        // Target transform: Rotate feature dim to back to be taken as inner-most axis
        // ex) 0(b), 4(f), 1(z), 2(y), 3(x)
        // ex) 0(b), 3(f), 1(y), 2(x)
        if ((int32_t) order[1] != order.size() - 1) return false;
        if ((int32_t) order[0] != 0) return false;
        for (int32_t i = 2; i < (int32_t) order.size(); ++i) {
            if ((int32_t)order[i] !=  (i - 1)) return false;
        }
        return true;
    };

    const permute_params& params = static_cast<const permute_params&>(p);

    if (params.inputs[0].GetDims().size() != params.output.GetDims().size()) {
        return false;
    }

    if (!is_rotating_except_batch(params.order)) {
        return false;
    }

    return true;
}

KernelsPriority PermuteKernel_tile_8x8_4x4_fsv::GetKernelsPriority(const Params& params, const optional_params& /*options*/) const {
    KernelData kd = KernelData::Default<permute_params>(params);
    permute_params& newParams = *static_cast<permute_params*>(kd.params.get());

    // calculate number of working groups
    const size_t tile_width = GetTileSize(newParams);
    const size_t tile_height = tile_width;
    const size_t fsv_alignment = GetFsvAlignment(newParams);

    std::vector<size_t> gws = GetGWS(newParams);
    std::vector<size_t> lws = GetBestLwsFromGws(newParams, gws, tile_width, tile_height);
    size_t num_working_groups = 1;
    for (size_t i=0; i < gws.size(); ++i) {
        num_working_groups *= gws.at(i)/lws.at(i);
    }

    const size_t feature = newParams.inputs[0].Feature().v;
    size_t rotating_dim = 0;
    if (newParams.inputs[0].GetDims().size() == 4) {
        rotating_dim = newParams.inputs[0].Y().v;
    } else if (newParams.inputs[0].GetDims().size() == 5) {
        rotating_dim = newParams.inputs[0].Z().v;
    }

    if (num_working_groups == 1) {
        return DONT_USE_IF_HAVE_SOMETHING_ELSE;
    } else if ((rotating_dim >= fsv_alignment) && (feature >= fsv_alignment)) {
        return FORCE_PRIORITY_1;
    } else if ((rotating_dim >= DEFAULT_TILE_SIZE) && (feature >= DEFAULT_TILE_SIZE)) {
        return FORCE_PRIORITY_2;
    } else if ((rotating_dim >= DEFAULT_TILE_SIZE) || (feature >= DEFAULT_TILE_SIZE)) {
        return FORCE_PRIORITY_3;
    } else {
        return DONT_USE_IF_HAVE_SOMETHING_ELSE;
    }
}
}  // namespace kernel_selector
