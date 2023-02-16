// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "configuration.h"
#include <algorithm>
#include "custom_task_arena.h"

namespace cldnn {
namespace gpu {

configuration::configuration()
    : enable_profiling(false),
      meaningful_kernels_names(false),
      dump_custom_program(false),
      host_out_of_order(true),
      use_unifed_shared_memory(false),
      compiler_options(""),
      single_kernel_name(""),
      log(""),
      ocl_sources_dumps_dir(""),
      priority_mode(priority_mode_types::disabled),
      throttle_mode(throttle_mode_types::disabled),
      queues_num(0),
      tuning_cache_path("cache.json"),
      kernels_cache_path(""),
      n_threads(std::max(static_cast<uint16_t>(std::thread::hardware_concurrency()), static_cast<uint16_t>(1))),
      core_type(cldnn::custom::task_arena::automatic) {
#if (CLDNN_THREADING == CLDNN_THREADING_TBB)
            if ((cldnn::custom::info::core_types().size() > 1 /*Hybrid CPUs*/)) {
                  n_threads = std::min(static_cast<uint16_t>(cldnn::custom::info::get_num_big_cores()), n_threads);
                  core_type = cldnn::custom::info::core_types().back(); // Now only big core supports
            }
#endif
      }
}  // namespace gpu
}  // namespace cldnn
