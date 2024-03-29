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

    virtual const std::shared_ptr<const NetworkDescription>& get_network_description() const = 0;

    virtual const Config& get_config() const = 0;

    // Compiler is used for post-processing profiling data when using PERF_COUNT property
    virtual const ov::SoPtr<ICompiler>& get_compiler() const = 0;

    const NetworkMetadata& get_network_metadata() const {
        return get_network_description()->metadata;
    }

protected:
    std::shared_ptr<const ICompiledModel> shared_from_this() const {
        return std::dynamic_pointer_cast<const ICompiledModel>(ov::ICompiledModel::shared_from_this());
    }
};

}  // namespace intel_npu
