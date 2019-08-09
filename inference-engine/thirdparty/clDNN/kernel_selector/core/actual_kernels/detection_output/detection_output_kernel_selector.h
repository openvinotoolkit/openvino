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

#include "kernel_selector.h"

namespace kernel_selector {
class detection_output_kernel_selector : public kernel_selector_base {
public:
    static detection_output_kernel_selector& Instance() {
        static detection_output_kernel_selector instance_;
        return instance_;
    }

    detection_output_kernel_selector();

    virtual ~detection_output_kernel_selector() {}

    KernelsData GetBestKernels(const Params& params, const optional_params& options) const override;
};

class detection_output_sort_kernel_selector : public kernel_selector_base {
public:
    static detection_output_sort_kernel_selector& Instance() {
        static detection_output_sort_kernel_selector instance_;
        return instance_;
    }

    detection_output_sort_kernel_selector();

    virtual ~detection_output_sort_kernel_selector() {}

    KernelsData GetBestKernels(const Params& params, const optional_params& options) const override;
};
}  // namespace kernel_selector