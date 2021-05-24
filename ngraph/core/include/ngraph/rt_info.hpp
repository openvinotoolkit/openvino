// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "ngraph/ngraph_visibility.hpp"
#include "ngraph/node.hpp"
#include "ngraph/type.hpp"

namespace ngraph
{
    NGRAPH_API
    void copy_runtime_info(std::shared_ptr<ngraph::Node> from, std::shared_ptr<ngraph::Node> to);

    NGRAPH_API
    void copy_runtime_info(std::shared_ptr<ngraph::Node> from, ngraph::NodeVector to);

    NGRAPH_API
    void copy_runtime_info(const ngraph::NodeVector& from, std::shared_ptr<ngraph::Node> to);

    NGRAPH_API
    void copy_runtime_info(const ngraph::NodeVector& from, ngraph::NodeVector to);
} // namespace ngraph
