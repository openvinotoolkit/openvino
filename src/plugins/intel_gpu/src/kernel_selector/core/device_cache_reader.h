// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <memory>
#include <string>

namespace kernel_selector {
class TuningCache;

std::shared_ptr<kernel_selector::TuningCache> CreateTuningCacheFromFile(std::string tuning_cache_path);

}  // namespace kernel_selector
