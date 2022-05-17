/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#ifndef COMPILER_WORKAROUNDS_HPP
#define COMPILER_WORKAROUNDS_HPP

// Workaround 01: clang.
//
// Clang has an issue [1] with `#pragma omp simd` that might lead to segfault.
// The essential conditions are:
//  1. Optimization level is O1 or O2. Surprisingly, O3 is fine.
//  2. Conditional check inside the vectorization loop.
// Since there is no reliable way to determine the first condition, we disable
// vectorization for clang altogether for now.
//
// [1] https://bugs.llvm.org/show_bug.cgi?id=48104
#if (defined __clang_major__) && (__clang_major__ >= 6)
#define CLANG_WA_01_SAFE_TO_USE_OMP_SIMD 0
#else
#define CLANG_WA_01_SAFE_TO_USE_OMP_SIMD 1
#endif

// Workaround 02: clang.
//
// Clang 6+ generates incorrect code with OMP_SIMD in some particular cases.
// Unlike CLANG_WA_01_SAFE_TO_USE_OMP_SIMD, the issue happens even with -O3.
#if (defined __clang_major__) && (__clang_major__ >= 6)
#define CLANG_WA_02_SAFE_TO_USE_OMP_SIMD 0
#else
#define CLANG_WA_02_SAFE_TO_USE_OMP_SIMD 1
#endif

// Workaround 03: GCC
//
// For very large functions with too much control flow (i.e. if, switch, goto
// statements), GCC 7 may struggle to perform optimizations based on tree
// dominator (i.e. -ftree-dominator-opts, which is enabled with O1), thereby
// producing an internal compiler error (ICE). Specifically, it seems that the
// jump threading optimization is the culprit, which cannot be disabled on its
// own. There is no reliable way to reproduce the ICE, therefore it is not clear
// which __GCC_MINOR__ version fixes issue.
#if (defined __GNUC__) && (__GNUC__ == 7) && (!defined(__INTEL_COMPILER)) \
        && (!defined(__clang__major__))
#define GCC_WA_NO_TREE_DOMINATOR_OPTS 1
#else
#define GCC_WA_NO_TREE_DOMINATOR_OPTS 0
#endif

// Workaround 04: GCC
//
// GCC 10 & 11 (at least versiona 10.1, 10.3 & 11.1) report false positives
// in xbyak when -Warray-bounds build setting is on
#if (!defined(__INTEL_COMPILER) && !defined(__clang__major__)) \
        && (defined(__GNUC__) && (__GNUC__ == 10 || __GNUC__ == 11))
#pragma GCC diagnostic ignored "-Warray-bounds"
#endif

#endif // COMPILER_WORKAROUNDS_HPP
