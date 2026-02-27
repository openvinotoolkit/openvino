// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/parser_factory.hpp"

#include "dynamic_parser.hpp"
#include "parser.hpp"

namespace intel_npu {

std::unique_ptr<IParser> ParserFactory::getParser(const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStructs,
                                                  ov::intel_npu::CompilerType& compilerType,
                                                  const ov::Tensor& blob) const {
    OPENVINO_ASSERT(
        zeroInitStructs != nullptr,
        "Could not find an NPU device. The driver compiler requires a valid device to be present in the system.");

#ifdef NPU_PLUGIN_DEVELOPER_BUILD
    const void* data = blob.data();
    size_t size = blob.get_byte_size();
    std::string header;
    if (size >= 20) {
        header.assign(static_cast<const char*>(data), 20);
    } else {
        header.assign(static_cast<const char*>(data), size);
    }
    if (header.find("ELF") == std::string::npos) {
        compilerType = ov::intel_npu::CompilerType::PLUGIN;
        return std::make_unique<DynamicParser>(zeroInitStructs);
    }
#endif

    return std::make_unique<Parser>(zeroInitStructs);
}

}  // namespace intel_npu
