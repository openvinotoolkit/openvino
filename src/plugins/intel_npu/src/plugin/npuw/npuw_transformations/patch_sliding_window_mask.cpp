// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "patch_sliding_window_mask.hpp"

#include "openvino/pass/manager.hpp"
#include "sliding_window_mask.hpp"

namespace {

bool patch_sliding_window_mask(const std::shared_ptr<ov::Model>& model) {
    ov::pass::Manager manager;
    manager.register_pass<ov::npuw::SlidingWindowMask>();
    return manager.run_passes(model);
}

}  // namespace

namespace ov::npuw {

bool PatchSlidingWindowMask::run_on_model(const std::shared_ptr<ov::Model>& model) {
    return patch_sliding_window_mask(model);
}

}  // namespace ov::npuw
