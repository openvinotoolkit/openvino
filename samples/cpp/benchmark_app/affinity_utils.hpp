// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "openvino/openvino.hpp"

void apply_manual_affinities(const std::shared_ptr<ov::Model>& model,
                             const std::string& affinity_spec,
                             const std::vector<std::string>& hardware_devices = {});