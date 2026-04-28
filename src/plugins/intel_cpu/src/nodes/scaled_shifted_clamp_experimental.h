// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <limits>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>

#include "graph_context.h"
#include "node.h"
#include "openvino/core/node.hpp"

namespace ov::intel_cpu::node {

class ScaledShiftedClamp : public Node {
public:
    ScaledShiftedClamp(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override {}
    void initSupportedPrimitiveDescriptors() override;
    void prepareParams() override {}
    void execute(const dnnl::stream& strm) override;
    void executeDynamicImpl(const dnnl::stream& strm) override {
        execute(strm);
    }
    [[nodiscard]] bool created() const override {
        return getType() == Type::ScaledShiftedClampExperimental;
    }
    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

private:
    float m_scale{1.0F};
    float m_bias{0.0F};
    float m_lo{std::numeric_limits<float>::lowest()};
    float m_hi{std::numeric_limits<float>::max()};
};

}  // namespace ov::intel_cpu::node
