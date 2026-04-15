// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "patch_phi3_sliding_mask.hpp"

#include "openvino/pass/manager.hpp"
#include "phi3_sliding_mask.hpp"

namespace {

bool patch_phi3_sliding_mask(const std::shared_ptr<ov::Model>& model) {
    ov::pass::Manager manager;
    manager.register_pass<ov::npuw::Phi3SlidingMask>();
    return manager.run_passes(model);
}

}  // namespace

namespace ov::npuw {

bool PatchPhi3SlidingMask::run_on_model(const std::shared_ptr<ov::Model>& model) {
    return patch_phi3_sliding_mask(model);
}

}  // namespace ov::npuw
