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

#ifdef _WIN32
#include "../../../common/core/cm/base_ops.hpp"
#include "../../../common/core/cm/base_types.hpp"
#include "../../../common/core/cm/common.hpp"
#include "../../../common/core/cm/math_general.hpp"
#else
#include "common/core/cm/base_ops.hpp"
#include "common/core/cm/base_types.hpp"
#include "common/core/cm/common.hpp"
#include "common/core/cm/math_general.hpp"
#endif

namespace gpu::xetla {

#define XETLA_PRINTF(s, ...) \
    do { \
    } while (0)

#define XETLA_ASSERT(c, s, ...) \
    do { \
    } while (0)

#define DEBUG_INVOKE(level, ...) \
    do { \
    } while (0)

} // namespace gpu::xetla