// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/any.hpp"

namespace ov {

class OPENVINO_API Meta {
public:
    virtual operator ov::AnyMap&() = 0;
    virtual operator const ov::AnyMap&() const = 0;
};

}  // namespace ov
