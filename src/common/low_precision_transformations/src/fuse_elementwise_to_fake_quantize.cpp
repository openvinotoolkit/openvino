// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/fuse_elementwise_to_fake_quantize.hpp"

#include <memory>
#include "low_precision/fake_quantize.hpp"
#include "low_precision/network_helper.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/fake_quantize.hpp"

namespace ov {
namespace pass {
namespace low_precision {

FuseElementwiseToFakeQuantizeTransformation::FuseElementwiseToFakeQuantizeTransformation(const Params& params) : CleanupTransformation(params) {
}

bool FuseElementwiseToFakeQuantizeTransformation::canBeTransformed(const std::shared_ptr<Node>& operation) const {
    if (!CleanupTransformation::canBeTransformed(operation)) {
        return false;
    }

    if (!ov::is_type<ov::op::v0::Constant>(operation->get_input_node_shared_ptr(1))) {
        return false;
    }

    if (!FakeQuantizeTransformation::checkElementwise(operation)) {
        return false;
    }

    const auto parent = operation->get_input_node_shared_ptr(0);
    auto fq = ov::as_type_ptr<ov::op::v0::FakeQuantize>(parent);
    const auto convert = ov::as_type_ptr<ov::op::v0::Convert>(parent);

    if (convert) {
        fq = ov::as_type_ptr<ov::op::v0::FakeQuantize>(convert->get_input_node_shared_ptr(0));
    }

    if (!fq) {
        return false;
    }

    if (fq->get_output_target_inputs(0).size() != 1) {
        return false;
    }

    return true;
}

} // namespace low_precision
} // namespace pass
} // namespace ov
