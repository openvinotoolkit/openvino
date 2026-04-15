// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>
#include <vector>

namespace ov::intel_cpu::node {

class PaKVReorder : public Node {
public:
    PaKVReorder(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override;

    void initSupportedPrimitiveDescriptors() override;

    void createPrimitive() override;

    void execute(const dnnl::stream& strm) override;

    void executeDynamicImpl(const dnnl::stream& strm) override;

    bool created() const override;

    bool needPrepareParams() const override {
        return false;
    }

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

private:
    // Quantization configuration (determined at createPrimitive time)
    bool m_key_by_channel = false;
    bool m_value_by_channel = false;
};

}  // namespace ov::intel_cpu::node
