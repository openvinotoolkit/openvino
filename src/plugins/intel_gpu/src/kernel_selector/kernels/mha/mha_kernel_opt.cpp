// Copyright (C) 2023-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mha_kernel_opt.h"
#include "kernel_selector_utils.h"
#include <string>
#include <vector>
#include <cmath>

namespace kernel_selector {
ParamsKey MHAKernelOpt::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F16);

    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);

    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    return k;
}

static std::pair<size_t, size_t> GetAvailableBlockSize(size_t N, size_t d, const mha_params& params) {
    size_t M = static_cast<size_t>(params.engineInfo.maxLocalMemSize/2);

    // initial block size
    size_t blk_col_size = static_cast<size_t>(std::ceil(M/(4 * d)));
    size_t blk_row_size = std::min(blk_col_size, d);

    /*
     * Used shared local memories
     *    Qi = Br x d
     *    Ki = Bc x d
     *    Vi = Bc x d
     *    Si = Br x Bc
     *    Oi = Br x d
     */
    while (true) {
        size_t estimated = 2 * (blk_row_size * d) + 2 * (blk_col_size * d) + blk_row_size * blk_col_size;
        if (estimated < M)
            break;
        if (blk_row_size > params.engineInfo.maxWorkGroupSize) {
            blk_row_size /= 2;
        } else {
            blk_col_size = blk_col_size/2;
        }
    }
    return std::make_pair(blk_row_size, blk_col_size);
}

MHAKernelOpt::TuningData MHAKernelOpt::GetTuningData(const mha_params& params) const {
    size_t N = params.inputs[0].Y().v;
    size_t d = params.inputs[0].X().v;

    auto block_size = GetAvailableBlockSize(N, d, params);

    TuningData td;
    td.num_words      = N;
    td.depth_size     = d;
    td.blk_row_size   = block_size.first;
    td.blk_col_size   = block_size.second;
    td.num_blk_row    = static_cast<size_t>(std::ceil(N/td.blk_row_size));
    td.num_blk_col    = static_cast<size_t>(std::ceil(N/td.blk_col_size));
    td.q_blk_size     = td.blk_row_size * d;
    td.k_blk_size     = td.blk_col_size * d;
    td.v_blk_size     = td.k_blk_size;
    td.score_mat_size = td.blk_row_size * td.blk_col_size;
    td.out_blk_size   = td.q_blk_size;

    return td;
}

CommonDispatchData MHAKernelOpt::SetDefault(const mha_params& params) const {
    CommonDispatchData dispatchData;

    auto batch = params.inputs[0].Batch().v;
    auto feature = params.inputs[0].Feature().v;

    /* FIXME: even for ref implementation, we can parallelize f-axis */
    auto td = GetTuningData(params);
    std::cout << __FILE__ << ":" << __LINE__ << " Tuning Data:\n" << td.to_string()
        << ", max group size: " << params.engineInfo.maxWorkGroupSize << std::endl;
    dispatchData.gws = {batch * feature, td.num_blk_row, td.blk_row_size};
    dispatchData.lws = {1, 1, td.blk_row_size};
    // dispatchData.gws = {batch * feature, td.num_blk_row, 16};
    // dispatchData.lws = {1, 1, 16};

    std::cout << __FILE__ << ":" << __LINE__ << " GWS: "
        << dispatchData.gws[0] << ", "
        << dispatchData.gws[1] << ", "
        << dispatchData.gws[2] << std::endl;

    std::cout << __FILE__ << ":" << __LINE__ << " LWS: "
        << dispatchData.lws[0] << ", "
        << dispatchData.lws[1] << ", "
        << dispatchData.lws[2] << std::endl;

    return dispatchData;
}

JitConstants MHAKernelOpt::GetJitConstants(const mha_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    auto td = GetTuningData(params);

    jit.AddConstants({
        MakeJitConstant("NUM_WORDS", td.num_words),
        MakeJitConstant("DEPTH_SIZE", td.depth_size),
        MakeJitConstant("BLK_ROW_SIZE", td.blk_row_size),
        MakeJitConstant("BLK_COL_SIZE", td.blk_col_size),
        MakeJitConstant("NUM_BLK_ROW", td.num_blk_row),
        MakeJitConstant("NUM_BLK_COL", td.num_blk_col),
        MakeJitConstant("Q_BLK_SIZE", td.q_blk_size),
        MakeJitConstant("K_BLK_SIZE", td.k_blk_size),
        MakeJitConstant("V_BLK_SIZE", td.v_blk_size),
        MakeJitConstant("SCORE_MAT_SIZE", td.score_mat_size),
        MakeJitConstant("OUT_BLK_SIZE", td.out_blk_size),
    });

    return jit;
}

bool MHAKernelOpt::Validate(const Params& p, const optional_params& o) const {
    if (p.GetType() != KernelType::MHA || o.GetType() != KernelType::MHA) {
        return false;
    }

    auto params = static_cast<const mha_params&>(p);

    /* FIXME: fill here to allow SD-2.1 only */
    // if (params.inputs[0].Feature().v != 10 ||
    //     params.inputs[0].Y().v != 9216 || params.inputs[0].X().v != 64) {
    //     std::cout << __FILE__ << ":" << __LINE__ << " " << "Failed to validation" << std::endl;
    //     return false;
    // }

    if (params.fused_ops.size() > 0)
        return false;

    return true;
}

KernelsData MHAKernelOpt::GetKernelsData(const Params& params, const optional_params& options) const {
    KernelData kd = KernelData::Default<mha_params>(params);
    mha_params& newParams = *static_cast<mha_params*>(kd.params.get());

    if (!Validate(params, options)) {
        return {};
    }

    auto dispatchData = SetDefault(newParams);
    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, params, options);
    auto cldnn_jit = GetJitConstants(newParams);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];

    FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point,
                     EXE_MODE_DEFAULT, false, false, 3, GetFusedPrimitiveInputsCount(params));

    return { kd };}

}  // namespace kernel_selector
