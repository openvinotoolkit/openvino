// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/jitter.hpp"
#include "utils/kernel_base.hpp"

namespace ov {
namespace intel_gpu {
namespace ocl {

struct SDPABase : public ov::intel_gpu::ocl::SingleKernelGenerator {
    SDPABase(const std::string name, bool indirect) : ov::intel_gpu::ocl::SingleKernelGenerator(name), m_indirect(indirect) {}
    JitConstants get_jit_constants(const program_node& node, const kernel_impl_params& params) const override;

    bool m_indirect;
};

}  // namespace ocl
}  // namespace intel_gpu
}  // namespace ov
