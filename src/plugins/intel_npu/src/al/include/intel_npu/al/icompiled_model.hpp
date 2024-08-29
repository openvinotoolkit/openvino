// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "intel_npu/al/config/common.hpp"
#include "intel_npu/al/icompiler.hpp"
#include "openvino/runtime/icompiled_model.hpp"

namespace intel_npu {

class ICompiledModel : public ov::ICompiledModel {
public:
    using ov::ICompiledModel::ICompiledModel;

    virtual const std::vector<uint8_t>& get_compiled_network() const = 0;

    virtual const Config& get_config() const = 0;

    // Compiler is used for post-processing profiling data when using PERF_COUNT property
    virtual const ov::SoPtr<ICompiler>& get_compiler() const = 0;

    virtual const NetworkMetadata& get_network_metadata() const = 0;

protected:
    std::shared_ptr<const ICompiledModel> shared_from_this() const {
        return std::dynamic_pointer_cast<const ICompiledModel>(ov::ICompiledModel::shared_from_this());
    }
};

}  // namespace intel_npu
