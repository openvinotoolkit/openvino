//
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <cstdint>
#include "meta.hpp"

namespace compat::tlv {

// see compat/requirements.hpp for ABI stability

#pragma pack(push, 1)

struct alignas(uint16_t) Header {
    Header(uint16_t type, bool required, uint8_t offset, uint16_t length);

    // uint16_t type allows for 65536 capability entries
    // which is probably enough; be cautious about headers size
    // as compatibility data size maybe sensitive
    uint16_t type;

    uint16_t required : 1;
    // TODO: revise this
    uint16_t offset   : 3;
    // uint16_t version;
    uint16_t length   : 12;
};

// TODO: add static_assert that size of Header is expected
static_assert(compat::meta::IsMemCopyable<Header>::value, "tlv::Header must be trivially copyable and of standard layout");

#pragma pack(pop)

}  // namespace compat::tlv
