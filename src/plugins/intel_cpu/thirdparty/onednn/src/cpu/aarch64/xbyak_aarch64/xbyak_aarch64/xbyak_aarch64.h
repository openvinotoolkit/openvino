#pragma once
/*******************************************************************************
 * Copyright 2019-2021 FUJITSU LIMITED
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

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#define NOMINMAX
#include <windows.h>
#undef mvn
#endif

#include <algorithm>
#include <deque>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <list>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if defined(__GNUC__) || defined(__APPLE__)
#ifndef XBYAK_USE_MMAP_ALLOCATOR
#define XBYAK_USE_MMAP_ALLOCATOR
#endif
#endif

#include <cmath>
#include <functional>

#if defined(__GNUC__)
#include <cassert>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

#include <iomanip>
#include <sstream>

#ifndef NDEBUG
#include <iostream>
#endif

#if defined(__APPLE__)
#define XBYAK_USE_MAP_JIT
#include <sys/sysctl.h>
#ifndef MAP_JIT
#define MAP_JIT 0x800
#endif
#endif

namespace Xbyak_aarch64 {
const uint64_t SP_IDX = 31;
const uint64_t NUM_VREG_BYTES = 16;
const uint64_t NUM_ZREG_BYTES = 64;
#include "xbyak_aarch64_gen.h"
} // namespace Xbyak_aarch64
