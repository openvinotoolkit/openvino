// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "graph_transformer.hpp"

#include <vpu/model/base.hpp>

namespace vpu {

CompiledGraph::Ptr compileModel(
        const Model& model,
        Platform platform,
        const CompilationConfig& config,
        const Logger::Ptr& log);

}  // namespace vpu
