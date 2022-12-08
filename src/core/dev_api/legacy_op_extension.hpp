// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/op_extension.hpp"

namespace ov {

/** @brief Class to distinguish legacy extension. */
class OPENVINO_API LegacyOpExtension : public BaseOpExtension {
public:
    ~LegacyOpExtension() override;
};
}  // namespace ov
