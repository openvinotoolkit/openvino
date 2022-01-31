// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "graph_transformer.hpp"

#include <vpu/model/base.hpp>

namespace vpu {

CompiledGraph::Ptr compileModel(
        const Model& model,
        const PluginConfiguration& config,
        const Logger::Ptr& log);

}  // namespace vpu
