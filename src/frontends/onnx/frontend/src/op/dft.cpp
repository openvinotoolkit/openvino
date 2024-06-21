// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/dft.hpp"

#include "core/null_node.hpp"
#include "core/operator_set.hpp"
#include "utils/common.hpp"
namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_1 {
ov::OutputVector dft(const ov::frontend::onnx::Node& node) {
    const ov::OutputVector ng_inputs{node.get_ov_inputs()};
    const ov::Output<ov::Node> data = ng_inputs.at(0);

    const auto dft_length_provided = ng_inputs.size() > 1 && !ov::op::util::is_null(ng_inputs[1]);
    const auto axis = node.get_attribute_value<int64_t>("axis", 1);
    const auto inverse = node.get_attribute_value<int64_t>("inverse", 0);
    const auto onesided = node.get_attribute_value<int64_t>("onesided", 0);

    return {dft::make_dft(data,
                          dft_length_provided ? ng_inputs.at(1) : std::make_shared<NullNode>(),
                          axis,
                          inverse == 1,
                          onesided == 1)};
}

static bool registered = register_translator("DFT", VersionRange::single_version_for_all_opsets(), dft);
}  // namespace set_1
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
