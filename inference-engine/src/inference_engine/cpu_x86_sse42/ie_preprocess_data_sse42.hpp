// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ie_blob.h"

#include <stdint.h>

namespace InferenceEngine {

void resize_bilinear_u8(const Blob::Ptr inBlob, Blob::Ptr outBlob, uint8_t* buffer);

void resize_area_u8_downscale(const Blob::Ptr inBlob, Blob::Ptr outBlob, uint8_t* buffer);

}
