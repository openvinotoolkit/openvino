// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief OpenVINO Runtime CompiledModel interface
 * @file icompiled_model.hpp
 */

#pragma once

#include <memory>
#include <openvino/runtime/common.hpp>
namespace ov {

class OPENVINO_RUNTIME_API ICompiledModel : public std::enable_shared_from_this<ICompiledModel> {
public:
    using Ptr = std::shared_ptr<ICompiledModel>;
};

}  // namespace ov
