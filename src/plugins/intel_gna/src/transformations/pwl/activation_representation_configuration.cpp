// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "activation_representation_configuration.hpp"

namespace ov {
namespace intel_gna {
namespace pass {
namespace pwl {

ActivationRepresentationConfigurationBase::ActivationRepresentationConfigurationBase(
    std::shared_ptr<BoundariesFQHandler> boundaries_fq_handler)
    : m_boundaries_fq_handler(std::move(boundaries_fq_handler)) {}

std::vector<FunctionSubinterval> ActivationRepresentationConfigurationBase::generate_subintervals(
    std::shared_ptr<ov::Node> fake_quantize,
    const SubintervalsCreator& creator,
    double lower_bound,
    double upper_bound) const {
    auto effective_lower_bound = lower_bound;
    auto effective_upper_bound = upper_bound;

    if (m_boundaries_fq_handler) {
        std::tie(effective_lower_bound, effective_upper_bound) =
            m_boundaries_fq_handler->get_adjust_boundaries({lower_bound, upper_bound}, fake_quantize);
    }

    return creator.generate(effective_lower_bound, effective_upper_bound);
}

}  // namespace pwl
}  // namespace pass
}  // namespace intel_gna
}  // namespace ov