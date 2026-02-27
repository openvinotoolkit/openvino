// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_npu/common/iparser.hpp"
#include "intel_npu/utils/zero/zero_init.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"

namespace intel_npu {

class ParserFactory final {
public:
    std::unique_ptr<IParser> getParser(const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStructs,
                                       ov::intel_npu::CompilerType& compilerType,
                                       const ov::Tensor& blob) const;
};

}  // namespace intel_npu
