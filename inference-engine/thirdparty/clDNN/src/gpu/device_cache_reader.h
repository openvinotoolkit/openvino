// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
