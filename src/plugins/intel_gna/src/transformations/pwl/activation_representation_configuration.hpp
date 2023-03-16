// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "activation_representation.hpp"
#include "boundaries_fq_hanlder.hpp"
#include "openvino/core/node.hpp"

namespace ov {
namespace intel_gna {
namespace pass {
namespace pwl {

class ActivationRepresentationConfiguration {
public:
    virtual ~ActivationRepresentationConfiguration() = default;
    virtual std::shared_ptr<ov::Node> get_pattern_node(std::shared_ptr<ov::Node> input) const = 0;

    virtual std::shared_ptr<ActivationRepresentation> create_representation(
        std::shared_ptr<ov::Node> node,
        std::shared_ptr<ov::Node> fake_quantize) const = 0;
};

class ActivationRepresentationConfigurationBase : public ActivationRepresentationConfiguration {
public:
    ActivationRepresentationConfigurationBase(std::shared_ptr<BoundariesFQHandler> boundaries_fq_handler);

    ~ActivationRepresentationConfigurationBase() override = default;

protected:
    std::vector<FunctionSubinterval> generate_subintervals(std::shared_ptr<ov::Node> fake_quantize,
                                                           const SubintervalsCreator& creator,
                                                           double lower_bound,
                                                           double upper_bound) const;

private:
    std::shared_ptr<BoundariesFQHandler> m_boundaries_fq_handler;
};

}  // namespace pwl
}  // namespace pass
}  // namespace intel_gna
}  // namespace ov