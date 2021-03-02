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


#include "permute_kernel_tile_8x8_4x4_fsv16.h"
#include "kernel_selector_utils.h"
#include <string>
#include <functional>
#include <cmath>
#include <iostream>

#define CEIL_DIV(A, B) ((A + B - 1)/(B))

static const size_t tile_width  = 8;
static const size_t tile_height = 8;
static const size_t fsv_length = 16;

namespace kernel_selector {

ParamsKey PermuteKernel_tile_8x8_4x4_fsv16::GetSupportedKey() const {
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
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    return k;
}

static inline std::vector<std::string> GetFusedOpOrderVector(size_t size) {
    std::vector<std::string> res;
    switch (size) {
        case 4 :
            res = {"b", "y", "(x * TILE_SIZE + i)", "(f * TILE_SIZE + lh)"};
            break;
        case 5 :
            res = {"b", "z", "y", "(x * TILE_SIZE + i)", "(f * TILE_SIZE + lh)"};
            break;
        case 6 :
            res = {"b", "w", "z", "y", "(x * TILE_SIZE + i)", "(f * TILE_SIZE + lh)"};
            break;
        default : throw std::runtime_error("Unsupported combination\n");
    }
    return res;

}

static inline std::string GetTiledOutputOrder(size_t size) {
    std::string order_str = "";
    switch (size) {
        case 4 :
            order_str = "b, y, x, f + lh";
            break;
        case 5 :
            order_str = "b, z, y, x, f + lh";
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


JitConstants PermuteKernel_tile_8x8_4x4_fsv16::GetJitConstants(const permute_params& params, const CommonDispatchData& dispatchData) const {
    auto jit = Parent::GetJitConstants(params, dispatchData);
    const uint32_t f = params.inputs[0].Feature().v;
    // const uint32_t y = params.inputs[0].Y().v;
    const uint32_t x = params.inputs[0].X().v;
    // Note: this is default mode and different vector width is not being used now.
    const uint64_t total_lws = dispatchData.lws[0] * dispatchData.lws[1] * dispatchData.lws[2];
    const uint64_t transpose_buffer_size = tile_width*tile_height * total_lws;

    jit.AddConstant(MakeJitConstant("INPUT0_TILED_ORDER", GetTiledInputOrder(params.inputs[0].GetDims().size())));
    jit.AddConstant(MakeJitConstant("OUTPUT_TILED_ORDER", GetTiledOutputOrder(params.output.GetDims().size())));

    jit.AddConstant(MakeJitConstant("INPUT0_SIZE_FS", CEIL_DIV(f, fsv_length)));
    jit.AddConstant(MakeJitConstant("INPUT0_SIZE_F", f));
    jit.AddConstant(MakeJitConstant("TILE_WIDTH", tile_width));
    jit.AddConstant(MakeJitConstant("TILE_HEIGHT", tile_height));
    jit.AddConstant(MakeJitConstant("TILE_STRIDE", fsv_length * x));
    jit.AddConstant(MakeJitConstant("FSV_LENGTH", fsv_length));
    jit.AddConstant(MakeJitConstant("GROUP_STRIDE", tile_width * x * fsv_length));
    jit.AddConstant(MakeJitConstant("INPUTVTYPE", "CAT(INPUT0_TYPE, TILE_WIDTH)"));
    jit.AddConstant(MakeJitConstant("OUTPUTVTYPE", "CAT(OUTPUT_TYPE, TILE_HEIGHT)"));
    jit.AddConstant(MakeJitConstant("VLOAD", "CAT(vload, TILE_WIDTH)"));
    jit.AddConstant(MakeJitConstant("VSTORE", "CAT(vstore, TILE_HEIGHT)"));
    jit.AddConstant(MakeJitConstant("AS_INPUTVTYPE", "CAT(as_, INPUTVTYPE)"));
    jit.AddConstant(MakeJitConstant("AS_OUTPUTVTYPE", "CAT(as_, OUTPUTVTYPE)"));
    jit.AddConstant(MakeJitConstant("LOCAL_BUF_STRIDE", tile_width*tile_height));
    jit.AddConstant(MakeJitConstant("TRANS_BUF_SIZE", transpose_buffer_size / tile_height));

    std::string normal_condition = "true";
    std::string f_remainder_condition = "true";
    if (f % fsv_length != 0) // when F is not fsv-aligned
    {
        normal_condition += "&& (f < (INPUT0_SIZE_F - F_REMAINDER_SIZE))";
        f_remainder_condition += "&& ((INPUT0_SIZE_F - F_REMAINDER_SIZE) <= f) && (f < INPUT0_SIZE_F)";
        jit.AddConstant(MakeJitConstant("F_REMAINDER_SIZE", f % tile_width));
    }
    jit.AddConstant(MakeJitConstant("NORMAL_CONDITION", normal_condition));
    jit.AddConstant(MakeJitConstant("F_REMAINDER_CONDITION", f_remainder_condition));

    if (!params.fused_ops.empty()) {
        std::vector<std::string> output_order = GetFusedOpOrderVector(params.output.GetDims().size());
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
    const size_t max_work_group_size= params.engineInfo.maxWorkGroupSize;
    size_t max_num_work_items = std::min(max_work_group_size, max_local_mem_size / (elem_size * tile_width * tile_height));

    for (size_t i = 0; i < dims.size(); ++i) {
        size_t dim = dims[i];
        size_t max_divider = static_cast<size_t>(std::sqrt(gws[dim]) + 1);
        for (size_t divider = 1; divider <= max_divider; ++divider)
        {
            if (gws[dim] % divider == 0)
            {
                const size_t lws0 = gws[dim] / divider;
                if (lws0 <= max_num_work_items)
                {
                    lws[dim] = std::max(lws[dim], lws0);
                }
                if (divider <= max_num_work_items)
                {
                    lws[dim] = std::max(lws[dim], divider);
                }
            }
        }
        max_num_work_items /= lws[dim];
    }
    return lws;
}

CommonDispatchData PermuteKernel_tile_8x8_4x4_fsv16::SetDefault(const permute_params& params) const {
    CommonDispatchData dispatchData;
    const auto& in =  params.inputs[0];
    switch (in.GetLayout()) {
        case DataLayout::b_fs_yx_fsv16:
            dispatchData.gws = {CEIL_DIV(fsv_length, tile_width),
                CEIL_DIV(in.X().v * in.Y().v, tile_height),
                in.Batch().v * CEIL_DIV(in.Feature().v, fsv_length)};
            break;
        case DataLayout::b_fs_zyx_fsv16:
            dispatchData.gws = {CEIL_DIV(fsv_length, tile_width),
                CEIL_DIV(in.X().v * in.Y().v * in.Z().v, tile_height),
                in.Batch().v * CEIL_DIV(in.Feature().v, fsv_length)};
            break;
        default:
            throw std::runtime_error("Unsupported combination\n");
            break;
    }

    dispatchData.lws = GetBestLwsFromGws(params, dispatchData.gws, tile_width, tile_height);

    std::cout << "-------------------------------------------\n";
    if (in.GetDims().size() == 4)
        std::cout << "(b, f, y, x): " << in.Batch().v << ' ' << in.Feature().v << ' ' << in.Y().v << ' ' << in.X().v << '\n';
    else if (in.GetDims().size() == 5)
        std::cout << "(b, f, z, y, x): " << in.Batch().v << ' ' << in.Feature().v << ' ' << in.Z().v << ' ' << in.Y().v << ' ' << in.X().v << '\n';
    std::cout << "GWS: ";
    for (auto& e: dispatchData.gws) std::cout << e << ' ';
    std::cout << '\n';
    std::cout << "LWS: ";
    for (auto& e: dispatchData.lws) std::cout << e << ' ';
    std::cout << '\n';
    std::cout << "Tile Width: " << tile_width << '\n';
    std::cout << "Tile Height: " << tile_height << '\n';
    std::cout << "FSV Alignment: " << fsv_length << '\n';
    std::cout << '\n';
    return dispatchData;
}

// Validate is the same as permute_kernel_tile_8x8_4x4
bool PermuteKernel_tile_8x8_4x4_fsv16::Validate(const Params& p, const optional_params& o) const {
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

    if (params.output.PitchesDifferFromLogicalDims() || params.inputs[0].PitchesDifferFromLogicalDims()) {
        return false;
    }

    return true;
}

// GetKernelsPriority is the same as permute_kernel_tile_8x8_4x4
KernelsPriority PermuteKernel_tile_8x8_4x4_fsv16::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    // KernelData kd = KernelData::Default<permute_params>(params);
    // permute_params& newParams = *static_cast<permute_params*>(kd.params.get());

    // return FORCE_PRIORITY_1;
    return DONT_USE_IF_HAVE_SOMETHING_ELSE * 2;
}
}  // namespace kernel_selector
