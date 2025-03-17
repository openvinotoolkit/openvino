/*******************************************************************************
* Copyright (c) 2022-2023 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#pragma once

/// @defgroup xetla_util XeTLA Utility
/// This is low level API wrapper for utility functions.
/// Including the tensor prefetch/load/store API based on xetla_raw_send, simd_lane id generation API and so on.

/// @addtogroup xetla_util
/// @{

/// @defgroup xetla_util_tensor_load_store Tensor load store API
/// Implements the tensor load store functionality using raw send instructions.

/// @defgroup xetla_util_misc Util misc API
/// Implements some useful and commonly used APIs, such as vector generation API.

/// @defgroup xetla_util_rand Random number generator API
/// Philox rng, will generate 4 uint32_t random number per call.

/// @defgroup xetla_util_group XeTLA Group
/// This is a group API to define operation scope.

/// @defgroup xetla_util_named_barrier Named barrier API
/// This is a raw_send based named barrier API.

/// @} xetla_util

#ifdef _WIN32
#include "../../../common/utils/cm/common.hpp"
#include "../../../common/utils/cm/dict.hpp"
#include "../../../common/utils/cm/fastmath.hpp"
#include "../../../common/utils/cm/memory_descriptor.hpp"
#include "../../../common/utils/cm/misc.hpp"
#include "../../../common/utils/cm/nd_item.hpp"
#include "../../../common/utils/cm/rand.hpp"
#include "../../../common/utils/cm/raw_send_load_store.hpp"
#include "../../../common/utils/cm/raw_send_nbarrier.hpp"
#include "../../../common/utils/cm/work_group.hpp"
#else
#include "common/utils/cm/common.hpp"
#include "common/utils/cm/dict.hpp"
#include "common/utils/cm/fastmath.hpp"
#include "common/utils/cm/memory_descriptor.hpp"
#include "common/utils/cm/misc.hpp"
#include "common/utils/cm/nd_item.hpp"
#include "common/utils/cm/rand.hpp"
#include "common/utils/cm/raw_send_load_store.hpp"
#include "common/utils/cm/raw_send_nbarrier.hpp"
#include "common/utils/cm/work_group.hpp"
#endif
