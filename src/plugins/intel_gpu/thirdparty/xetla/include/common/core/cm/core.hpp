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

/// @file
/// C++ API

#pragma once

/// @defgroup xetla_core XeTLA Core

/// Some terminologies used in the API documentation:
/// - *word* - 2 bytes.
/// - *dword* ("double word") - 4 bytes.
/// - *qword* ("quad word") - 8 bytes.
/// - *oword* ("octal word") - 16 bytes.
///

/// @addtogroup xetla_core
/// @{

/// @defgroup xetla_core_base_types Base types
/// Defines vector, vector reference and matrix reference data types.

/// @defgroup xetla_core_base_ops Base ops
/// Defines base ops for vector, vector reference and matrix reference data types.

/// @defgroup xetla_core_memory Memory access APIs
/// Defines XeTLA APIs to access memory, including read, write and atomic.

/// @defgroup xetla_core_barrier Synchronization APIs
/// Defines XeTLA APIs for synchronization primitives.

/// @defgroup xetla_core_math Math operation APIs
/// Defines math operations on XeTLA vector data types.

/// @defgroup xetla_core_bit_manipulation Bit and mask manipulation APIs
/// Defines bitwise operations.

/// @defgroup xetla_core_conv Explicit conversion APIs
/// Defines explicit conversions (with and without saturation), truncation etc. between XeTLA vector types.

/// @defgroup xetla_core_raw_send Raw send APIs
/// Implements the \c send instruction to send messages to variaous components
/// of the Intel(R) processor graphics, as defined in the documentation at [here](https://01.org/sites/default/files/documentation/intel-gfx-prm-osrc-icllp-vol02a-commandreference-instructions_2.pdf)

/// @defgroup xetla_core_misc Miscellaneous XeTLA convenience functions
/// Wraps some useful functions.

/// @defgroup xetla_core_arch_config Arch config information
/// Defines some hardware arch related information, mainly used to do HW limitation check.

/// @} xetla_core

#ifdef _WIN32
#include "../../../common/core/cm/arch_config.hpp"
#include "../../../common/core/cm/barrier.hpp"
#include "../../../common/core/cm/base_consts.hpp"
#include "../../../common/core/cm/base_ops.hpp"
#include "../../../common/core/cm/base_types.hpp"
#include "../../../common/core/cm/bit_mask_manipulation.hpp"
#include "../../../common/core/cm/debug.hpp"
#include "../../../common/core/cm/explicit_conv.hpp"
#include "../../../common/core/cm/math_fma.hpp"
#include "../../../common/core/cm/math_general.hpp"
#include "../../../common/core/cm/math_mma.hpp"
#include "../../../common/core/cm/memory.hpp"
#include "../../../common/core/cm/misc.hpp"
#include "../../../common/core/cm/raw_send.hpp"
#else
#include "common/core/cm/arch_config.hpp"
#include "common/core/cm/barrier.hpp"
#include "common/core/cm/base_consts.hpp"
#include "common/core/cm/base_ops.hpp"
#include "common/core/cm/base_types.hpp"
#include "common/core/cm/bit_mask_manipulation.hpp"
#include "common/core/cm/debug.hpp"
#include "common/core/cm/explicit_conv.hpp"
#include "common/core/cm/math_fma.hpp"
#include "common/core/cm/math_general.hpp"
#include "common/core/cm/math_mma.hpp"
#include "common/core/cm/memory.hpp"
#include "common/core/cm/misc.hpp"
#include "common/core/cm/raw_send.hpp"
#endif
