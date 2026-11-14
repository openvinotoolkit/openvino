// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <nlohmann/json.hpp>

#include "openvino/core/model.hpp"
#include "openvino/frontend/gguf/extension/weight_normalization.hpp"

namespace ov_gguf_test {

std::shared_ptr<ov::frontend::gguf::WeightNormalizationExtension> structural_weight_extension();
nlohmann::json graph_snapshot(const std::shared_ptr<ov::Model>& model);
int graph_snapshot_main(int argc, char** argv);

}  // namespace ov_gguf_test
