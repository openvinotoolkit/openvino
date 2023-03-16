// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <memory>
#include <string>
#include <vector>

#include "function.hpp"
#include "openvino/core/node.hpp"
#include "segment.hpp"
#include "subinterval_creator.hpp"
#include "surrounding_segments_inserter.hpp"

namespace ov {
namespace intel_gna {
namespace pass {
namespace pwl {

class BoundariesFQHandler {
public:
    virtual ~BoundariesFQHandler() = default;
    virtual std::pair<double, double> get_adjust_boundaries(std::pair<double, double> boundaries,
                                                            std::shared_ptr<ov::Node> fake_quantize) const {
        // For Log the FakeQuantize is omitted.
        return boundaries;
    };
};

class BoundariesFQHandlerImpl : public BoundariesFQHandler {
public:
    BoundariesFQHandlerImpl(bool prefer_fake_quantized = false);
    virtual std::pair<double, double> get_adjust_boundaries(std::pair<double, double> boundaries,
                                                            std::shared_ptr<ov::Node> fake_quantize) const override;

private:
    bool m_prefer_fake_quntized;
};

}  // namespace pwl
}  // namespace pass
}  // namespace intel_gna
}  // namespace ov