// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/parser_factory.hpp"

#include "parser.hpp"

namespace intel_npu {

std::unique_ptr<IParser> ParserFactory::getParser(const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStructs) const {
    OPENVINO_ASSERT(
        zeroInitStructs != nullptr,
        "Could not find an NPU device. The driver compiler requires a valid device to be present in the system.");

    return std::make_unique<Parser>(zeroInitStructs);
}

}  // namespace intel_npu
