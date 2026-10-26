//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifndef NPU_VM_EXPORT
#if defined(_WIN32)
#define NPU_VM_EXPORT __declspec(dllexport)
#else
#define NPU_VM_EXPORT __attribute__((visibility("default")))
#endif  // defined(_WIN32)
#endif  // NPU_VM_EXPORT
