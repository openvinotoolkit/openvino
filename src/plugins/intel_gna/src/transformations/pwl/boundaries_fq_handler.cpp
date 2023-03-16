// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "boundaries_fq_hanlder.hpp"
#include "common/graph_utils.hpp"
#include "log/debug.hpp"
#include "openvino/opsets/opset11.hpp"

namespace ov {
namespace intel_gna {
namespace pass {
namespace pwl {

BoundariesFQHandlerImpl::BoundariesFQHandlerImpl(bool prefer_fake_quantized)
    : m_prefer_fake_quntized(prefer_fake_quantized) {}

std::pair<double, double> BoundariesFQHandlerImpl::get_adjust_boundaries(
    std::pair<double, double> boundaries,
    std::shared_ptr<ov::Node> fake_quantize) const {
    auto fake_quantize_inst = std::dynamic_pointer_cast<ov::opset11::FakeQuantize>(fake_quantize);
    if (!fake_quantize_inst) {
        return boundaries;
    }

    double lower_bound = boundaries.first;
    double upper_bound = boundaries.second;

    auto input_low = std::dynamic_pointer_cast<ov::opset11::Constant>(fake_quantize_inst->get_input_node_shared_ptr(1));
    auto input_high =
        std::dynamic_pointer_cast<ov::opset11::Constant>(fake_quantize_inst->get_input_node_shared_ptr(2));

    double retrieved_lower_bound = 0.0;
    double retrieved_upper_bound = 0.0;
    if (!graph_utils::get_constant_value(input_low, retrieved_lower_bound) ||
        !graph_utils::get_constant_value(input_high, retrieved_upper_bound)) {
        THROW_GNA_EXCEPTION << "The unsupported type of element.";
    }

    if (m_prefer_fake_quntized) {
        // For Power FakeQuantize values are taken without comparing to bounds.
        return {retrieved_lower_bound, retrieved_upper_bound};
    }

    auto abs_max = std::max(std::fabs(std::min(retrieved_lower_bound, retrieved_upper_bound) * 1.25),
                            std::fabs(std::max(retrieved_lower_bound, retrieved_upper_bound) * 1.25));
    if (abs_max < std::abs(lower_bound)) {
        lower_bound = -abs_max;
    }

    if (abs_max < std::abs(upper_bound)) {
        upper_bound = abs_max;
    }

    return {lower_bound, upper_bound};
}
}  // namespace pwl
}  // namespace pass
}  // namespace intel_gna
}  // namespace ov