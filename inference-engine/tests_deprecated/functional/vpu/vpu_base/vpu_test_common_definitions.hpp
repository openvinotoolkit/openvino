// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <ie_blob.h>

enum class IRVersion { v7, v10 };

using IN_OUT_desc = std::vector<InferenceEngine::SizeVector>;

using WeightsBlob = InferenceEngine::TBlob<uint8_t>;
