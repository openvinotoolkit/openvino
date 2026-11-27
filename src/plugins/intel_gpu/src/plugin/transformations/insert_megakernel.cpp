// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "insert_megakernel.hpp"

#include <cstdlib>

#include "../../megakernel_interface/iOVIntegration.h"

namespace ov::intel_gpu {

bool InsertMegaKernel::run_on_model(const std::shared_ptr<ov::Model>& m) {
    if (const char* off = std::getenv("OV_MEGAKERNEL_DISABLE"); off && off[0] == '1') {
        return false;
    }
    return mk::InsertMegakernelTransformation(m);
}

}  // namespace ov::intel_gpu
