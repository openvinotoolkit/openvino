// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <numeric>

#include "core/node.hpp"
#include "utils/common.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace variadic {
/// \brief Create an OpenVINO version of an ONNX variadic operation.
///        This creates a subgraph with a series of binary operations.
///
/// \param node Incoming ONNX opearation.
///
/// \tparam T   Class of an OpenVINO binary operation (e.g. Add, Minimum, Maximum)
///
/// \return OpenVINO node equivalent of the ONNX operation

template <class T>
inline ov::OutputVector make_ng_variadic_op(
    const Node& node,
    const ov::op::AutoBroadcastSpec& auto_broadcast = ov::op::AutoBroadcastType::NUMPY) {
    const ov::OutputVector ng_inputs{node.get_ov_inputs()};

    // Templated binary operation - Creates Add, Minimum, Maximum, etc.
    const auto binary_operation = [&auto_broadcast](const ov::Output<ov::Node>& arg0,
                                                    const ov::Output<ov::Node>& arg1) {
        return std::make_shared<T>(arg0, arg1, auto_broadcast);
    };

    // Create a result node as a series of binary operations
    auto result = std::accumulate(std::next(std::begin(ng_inputs)),  // First operand value - the second input
                                  std::end(ng_inputs),               // Last value - final input
                                  ng_inputs.front(),                 // Initial value - first input
                                  binary_operation);

    if (ng_inputs.size() == 1) {
        common::mark_as_optimized_out(result);
    }

    return {result};
}

}  // namespace variadic
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
