// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <vector>

#include "openvino/core/node.hpp"
#include "openvino/core/symbol.hpp"

namespace ov::intel_gpu::mlir {

using NodePtr = std::shared_ptr<ov::Node>;
using SymbolPtr = std::shared_ptr<ov::Symbol>;
using InputVector = std::vector<ov::Input<ov::Node>>;

// Physical dimension order the compiled program expects for some inputs (key == input index).
// Inputs without an entry are plain (row-major). E.g.:
//   input_orders[1] = {0, 2, 1};  // input 1 is a logical [E, N, G] tensor stored in memory as [E, G, N]
using InputOrders = std::map<size_t, std::vector<size_t>>;

}  // namespace ov::intel_gpu::mlir