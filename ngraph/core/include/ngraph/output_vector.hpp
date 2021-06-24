// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

namespace ngraph
{
    class Node;

    template <typename T>
    class Output;

    using NodeVector = std::vector<std::shared_ptr<Node>>;
    using OutputVector = std::vector<Output<Node>>;
} // namespace ngraph
