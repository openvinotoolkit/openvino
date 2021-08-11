// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "ngraph/ngraph_visibility.hpp"
#include "ngraph/node.hpp"
#include "ngraph/type.hpp"

#include "ngraph/ngraph_namespace.hpp"

namespace ov
{
    NGRAPH_API
    void copy_runtime_info(std::shared_ptr<ov::Node> from, std::shared_ptr<ov::Node> to);

    NGRAPH_API
    void copy_runtime_info(std::shared_ptr<ov::Node> from, ov::NodeVector to);

    NGRAPH_API
    void copy_runtime_info(const ov::NodeVector& from, std::shared_ptr<ov::Node> to);

    NGRAPH_API
    void copy_runtime_info(const ov::NodeVector& from, ov::NodeVector to);
} // namespace ov
