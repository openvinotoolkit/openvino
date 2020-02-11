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

namespace cldnn {
namespace gpu {

class device_cache_reader {
public:
    device_cache_reader(const std::string tuning_file_path, size_t compute_units_count);
    std::shared_ptr<rapidjson::Document> get() { return _dev_cache; }
private:
    std::shared_ptr<rapidjson::Document> _dev_cache;
};

}  // namespace gpu
}  // namespace cldnn
