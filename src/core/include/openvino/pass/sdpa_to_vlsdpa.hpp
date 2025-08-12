// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/pass/pass.hpp"

namespace ov {
namespace pass {
/**
 * @brief The transformation replaces SDPA in ViTs by VLSDPA operation.
 * The input "attention_mask" is replaced by "accumulated sequence lengths".
 * Please note -
 * 1. This pass applies to QWen2.x-VL models only, which relies on user (genai) to set
 * rt_info of "model_type_hint".
 * 2. The pass will change model inputs w.r.t input names, shape, and data type. Therefore,
 * it should be applied at the beginning of transformation pipeline.
 * \ingroup ov_pass_cpp_api
 */
class OPENVINO_API SDPAToVLSDPA : public ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("SDPAToVLSDPA");

    explicit SDPAToVLSDPA();
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};
}  // namespace pass
}  // namespace ov
