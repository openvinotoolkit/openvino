// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "int_backend_visibility.hpp"
#include "ngraph/node.hpp"

namespace ov
{
    namespace runtime
    {
        namespace interpreter
        {
            using EvaluatorsMap = std::map<ov::NodeTypeInfo,
                                           std::function<bool(const std::shared_ptr<ov::Node>& node,
                                                              const ov::HostTensorVector& outputs,
                                                              const ov::HostTensorVector& inputs)>>;
            EvaluatorsMap& get_evaluators_map();
        } // namespace interpreter
    }     // namespace runtime
} // namespace ov
