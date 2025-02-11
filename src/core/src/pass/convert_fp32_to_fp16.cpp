// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/convert_fp32_to_fp16.hpp"

#include "openvino/cc/pass/itt.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/convert_precision.hpp"

bool ov::pass::ConvertFP32ToFP16::run_on_model(const std::shared_ptr<ov::Model>& f) {
    RUN_ON_MODEL_SCOPE(ConvertFP32ToFP16);
    ov::pass::Manager m(get_pass_config(), "ConvertFP32ToFP16");
    m.register_pass<ov::pass::ConvertPrecision>(precisions_map{{ov::element::f32, ov::element::f16}});
    m.run_passes(f);
    return false;
}
