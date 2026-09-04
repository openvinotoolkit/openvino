// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace ov::intel_cpu::kernel {

struct PagedSelectiveSSMJitRuntimeArgs;

void validate_paged_selective_ssm_jit_metadata(const PagedSelectiveSSMJitRuntimeArgs& args);

}  // namespace ov::intel_cpu::kernel
