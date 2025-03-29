// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/validate.hpp"

#include "openvino/cc/pass/itt.hpp"

bool ov::pass::Validate::run_on_model(const std::shared_ptr<ov::Model>& m) {
    RUN_ON_MODEL_SCOPE(Validate);
    m->validate_nodes_and_infer_types();
    return false;
}
