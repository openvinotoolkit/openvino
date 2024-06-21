// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "utils/arg_min_max_factory.hpp"
namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_1 {
ov::OutputVector argmax(const ov::frontend::onnx::Node& node) {
    const utils::ArgMinMaxFactory arg_factory(node);
    return {arg_factory.make_arg_max()};
}

static bool registered = register_translator("ArgMax", VersionRange{1, 11}, argmax);
}  // namespace set_1

namespace set_12 {
ov::OutputVector argmax(const ov::frontend::onnx::Node& node) {
    const utils::ArgMinMaxFactory arg_factory(node);
    return {arg_factory.make_arg_max()};
}

static bool registered = register_translator("ArgMax", VersionRange::since(12), argmax);
}  // namespace set_12
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
