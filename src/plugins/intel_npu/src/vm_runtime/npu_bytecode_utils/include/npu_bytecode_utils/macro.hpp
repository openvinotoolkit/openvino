//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifndef __has_builtin
#define __has_builtin(x) 0
#endif

// NOLINTBEGIN(cppcoreguidelines-macro-usage)

#if __has_builtin(__builtin_expect) || defined(__GNUC__)
#define NPU_VM_UNLIKELY(EXPR) __builtin_expect((bool)(EXPR), false)
#else
#define NPU_VM_UNLIKELY(EXPR) (EXPR)
#endif

// NOLINTEND(cppcoreguidelines-macro-usage)
