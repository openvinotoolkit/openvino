// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

// Shared registry of experimental ops. Each new experimental op appends its header here.
// Entire body is gated by ENABLE_EXPERIMENTAL_OPSET so downstream TUs that do not set
// the define see no experimental symbols.

#ifdef ENABLE_EXPERIMENTAL_OPSET

#    include "openvino/op/scaled_shifted_clamp_experimental.hpp"

#endif  // ENABLE_EXPERIMENTAL_OPSET
