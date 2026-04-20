// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "openvino/core/model.hpp"
#include "openvino/core/node.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace op {
namespace util {

TRANSFORMATIONS_API std::shared_ptr<ov::Model> extract_subgraph(
    const std::shared_ptr<ov::Model>& model,
    const std::vector<ov::Input<ov::Node>>& subgraph_inputs,
    const std::vector<ov::Output<ov::Node>>& subgraph_outputs);

TRANSFORMATIONS_API std::shared_ptr<ov::Model> extract_subgraph(
    const std::shared_ptr<ov::Model>& model,
    const std::multimap<std::string, size_t>& subgraph_inputs,
    const std::multimap<std::string, size_t>& subgraph_outputs);

}  // namespace util
}  // namespace op
}  // namespace ov
