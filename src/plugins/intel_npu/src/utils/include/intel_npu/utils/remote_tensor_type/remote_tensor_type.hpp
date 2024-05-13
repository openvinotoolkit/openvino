// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace intel_npu {

enum class RemoteMemoryType { EMPTY, L0_INTERNAL_BUF, SHARED_BUF };

enum class RemoteTensorType { INPUT, OUTPUT, BINDED };

}  // namespace intel_npu
