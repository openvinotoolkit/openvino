// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/parser_factory.hpp"

#include "parser.hpp"

namespace intel_npu {

std::unique_ptr<IParser> ParserFactory::getParser(const ov::SoPtr<IEngineBackend>& engineBackend) const {
    OPENVINO_ASSERT(engineBackend != nullptr,
                    "Could not find a NPU device. Using parser requires a valid device to be present in the system.");

    return std::make_unique<Parser>(engineBackend->getInitStructs());
}

}  // namespace intel_npu
