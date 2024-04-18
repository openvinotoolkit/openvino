// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/sdpa_to_paged_attention.hpp"

#include "openvino/cc/pass/itt.hpp"


bool ov::pass::SDPAToPagedAttention::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(SDPAToPagedAttention);

    // писать сюда
    std::cout << "[ STUB ] ov::pass::SDPAToPagedAttention::run_on_model\n";

    return true;
}
