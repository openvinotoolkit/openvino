// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>


#define FLOAT_TO_INT8(a) static_cast<int8_t>(((a) < 0)?((a) - 0.5f):((a) + 0.5f))
#define FLOAT_TO_INT16(a) static_cast<int16_t>(((a) < 0)?((a) - 0.5f):((a) + 0.5f))
#define FLOAT_TO_INT32(a) static_cast<int32_t>(((a) < 0)?((a)-0.5f):((a)+0.5f))
#define FLOAT_TO_INT64(a) static_cast<int64_t>(((a) < 0)?((a)-0.5f):((a)+0.5f))