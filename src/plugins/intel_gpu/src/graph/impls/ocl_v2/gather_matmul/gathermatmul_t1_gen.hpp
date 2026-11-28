// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../utils/kernel_generator.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"

namespace ov::intel_gpu::ocl {

// GatherMatmul over base-3 ternary (t1) expert weights: one subgroup per output column, token and expert slot.
class GatherMatmulT1Generator : public KernelGenerator {
public:
    // quant_only: the shared activation-quantize stage, one work-group per activation row
    explicit GatherMatmulT1Generator(bool quant_only = false) : KernelGenerator("gathermatmul_t1", quant_only ? "_t1q" : "_t1"), m_quant_only(quant_only) {}

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override;
    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override;
    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override;

private:
    bool m_quant_only = false;
};

// dp4a kernel: each activation row is quantized once in a separate stage into internal buffers 20 (int8 permuted,
// uint), 21 (scales, f32), 22 (group sums, i32) instead of in every work-group.
bool gathermatmul_t1_shared_quant(const RuntimeParams& params);
// uints of buffer 20 per activation row (an upper bound over the weight formats)
size_t gathermatmul_t1_shared_xq_uints(const RuntimeParams& params);
// groups per row (buffers 21 and 22)
size_t gathermatmul_t1_shared_groups(const RuntimeParams& params);

}  // namespace ov::intel_gpu::ocl
