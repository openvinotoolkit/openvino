// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "node.h"

namespace ov::intel_cpu::node {

class CybersporeTSSN : public Node {
public:
    CybersporeTSSN(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void prepareParams() override;

    void execute(const dnnl::stream& strm) override;
    void executeDynamicImpl(const dnnl::stream& strm) override;

    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

private:
    template <typename SelectiveT>
    void compute(const SelectiveT* selective_ptr, size_t selective_count);

    size_t m_work_amount = 0ULL;
    float m_homeostatic_setpoint = 0.0f;
    float m_decay_rate = 1.0f;
};

}  // namespace ov::intel_cpu::node
