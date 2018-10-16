/*
// Copyright (c) 2016 Intel Corporation
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
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "ocl_toolkit.h"

namespace cldnn {
    namespace gpu {

        configuration::configuration()
            : enable_profiling(false)
            , meaningful_kernels_names(false)
            , device_type(gpu)
            , device_vendor(0x8086)
            , compiler_options("")
            , single_kernel_name("")
            , host_out_of_order(false)
            , log("")
            , ocl_sources_dumps_dir("")
        {}
    }
}
