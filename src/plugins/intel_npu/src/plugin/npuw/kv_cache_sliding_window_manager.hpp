// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace ov {
namespace npuw {
namespace util {

// rt_info key stamped on SWA-managed past_key_values Parameters whose seq_len
// axis was shrunk to the SWA window size.
static constexpr const char* NPUW_KV_CACHE_SLIDING_RT_KEY = "npuw_kv_cache_sliding";

}  // namespace util
}  // namespace npuw
}  // namespace ov
