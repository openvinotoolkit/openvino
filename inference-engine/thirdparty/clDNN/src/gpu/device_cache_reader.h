// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <memory>
#include "document.h"
#include <string>

namespace kernel_selector {
class TuningCache;
}

namespace cldnn {
namespace gpu {

class device_cache_reader {
public:
    explicit device_cache_reader(const std::string tuning_file_path);
    std::shared_ptr<kernel_selector::TuningCache> get() { return _dev_cache; }

private:
    std::shared_ptr<kernel_selector::TuningCache> _dev_cache;
};

}  // namespace gpu
}  // namespace cldnn
