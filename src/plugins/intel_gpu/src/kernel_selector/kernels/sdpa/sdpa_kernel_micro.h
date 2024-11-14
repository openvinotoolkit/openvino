// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef ENABLE_ONEDNN_FOR_GPU

#pragma once

#include "sdpa_kernel_base.h"
#include "micro_utils.hpp"

namespace kernel_selector {
class SDPAKernelMicro : public SDPAKernelBase {
public:
    using Parent = SDPAKernelBase;
    SDPAKernelMicro() : SDPAKernelBase("sdpa_micro") {}
    virtual ~SDPAKernelMicro() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    bool Validate(const Params& p) const override;
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;
    CommonDispatchData SetDefault(const sdpa_params& params, const micro::Package& gemm_kq, const micro::Package& gemm_vs) const;
    JitConstants GetJitConstants(const sdpa_params& params, const micro::Package& gemm_kq, const micro::Package& gemm_vs) const;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return {};
    }

    void init_microkernels(const sdpa_params& params, micro::Package& gemm_kq, micro::Package& gemm_vs, bool is_prefill) const;
    clKernelData get_kernel_data(const sdpa_params& params, bool is_prefill) const;

private:
    static constexpr size_t prefill_id = 0;
    static constexpr size_t generate_id = 1;

    static constexpr size_t kq_id = 0;
    static constexpr size_t vs_id = 1;

    static std::mutex m;
};
}  // namespace kernel_selector


#endif // ENABLE_ONEDNN_FOR_GPU
