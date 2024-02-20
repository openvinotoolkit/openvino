// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "quantization_info.hpp"

bool ov::frontend::tensorflow_lite::QuantizationInfo::is_copyable() const {
    return false;
}