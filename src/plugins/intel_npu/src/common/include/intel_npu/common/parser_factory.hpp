// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_npu/common/iparser.hpp"
#include "intel_npu/common/npu.hpp"

namespace intel_npu {

class ParserFactory final {
public:
    std::unique_ptr<IParser> getParser(const ov::SoPtr<IEngineBackend>& engineBackend) const;
};

}  // namespace intel_npu
