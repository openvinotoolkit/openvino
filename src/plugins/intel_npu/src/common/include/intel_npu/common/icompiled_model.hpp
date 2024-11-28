// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "intel_npu/common/igraph.hpp"
#include "intel_npu/config/common.hpp"
#include "openvino/runtime/icompiled_model.hpp"

namespace intel_npu {

class ICompiledModel : public ov::ICompiledModel {
public:
    using ov::ICompiledModel::ICompiledModel;

    virtual const std::shared_ptr<IGraph>& get_graph() const = 0;

    virtual const Config& get_config() const = 0;

protected:
    std::shared_ptr<const ICompiledModel> shared_from_this() const {
        return std::dynamic_pointer_cast<const ICompiledModel>(ov::ICompiledModel::shared_from_this());
    }
};

}  // namespace intel_npu
