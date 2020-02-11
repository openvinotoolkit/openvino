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
#include "configuration.h"

namespace cldnn {
namespace gpu {

configuration::configuration()
    : enable_profiling(false),
      meaningful_kernels_names(false),
      dump_custom_program(false),
      host_out_of_order(true),
      compiler_options(""),
      single_kernel_name(""),
      log(""),
      ocl_sources_dumps_dir(""),
      priority_mode(priority_mode_types::disabled),
      throttle_mode(throttle_mode_types::disabled),
      queues_num(0),
      tuning_cache_path("cache.json") {}
}  // namespace gpu
}  // namespace cldnn
