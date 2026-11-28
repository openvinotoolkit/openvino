// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "intel_npu/common/icontext.hpp"
#include "intel_npu/common/iparser.hpp"
#include "intel_npu/utils/vcl/vcl_function_table_provider.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"

namespace intel_npu {

class ParserFactory final {
public:
    std::unique_ptr<IParser> getParser(const std::shared_ptr<IContext>& context,
                                       const VCLFunctionTableProvider& vclFunctionsProvider,
                                       bool perfCountEnabled) const;
};

}  // namespace intel_npu
