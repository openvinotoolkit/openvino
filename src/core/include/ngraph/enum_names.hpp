// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#warning("The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")

#include "ngraph/check.hpp"
#include "openvino/core/enum_names.hpp"

namespace ngraph {
using ov::as_enum;
using ov::as_string;
using ov::EnumNames;
}  // namespace ngraph
