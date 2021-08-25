// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

namespace ov {

template <typename NodeType>
class Output;

class Node;

using NodeVector = std::vector<std::shared_ptr<Node>>;
using OutputVector = std::vector<Output<Node>>;
}  // namespace ov
