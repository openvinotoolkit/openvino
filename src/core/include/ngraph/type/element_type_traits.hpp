// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#warning("The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")

#include "openvino/core/type/element_type_traits.hpp"

namespace ngraph {
using ov::element_type_traits;

using ov::fundamental_type_for;

}  // namespace ngraph
