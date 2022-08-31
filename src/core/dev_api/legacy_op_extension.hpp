// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/op_extension.hpp"

namespace ov {

/** @brief Class to distinguish legacy extension. */
class LegacyOpExtension : public BaseOpExtension {};
}  // namespace ov
