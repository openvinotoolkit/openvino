//
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include <cstdint>
#include "tlv.hpp"

namespace compat::tlv {

Header::Header(uint16_t type, bool required, uint8_t offset, uint16_t length) :
    type(type), required(required), offset(offset), length(length) {}

}  // namespace compat::tlv
