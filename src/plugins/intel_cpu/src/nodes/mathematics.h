// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "node.h"

namespace ov {
namespace intel_cpu {
namespace node {

class Math : public Node {
public:
    Math(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override{};
    void initSupportedPrimitiveDescriptors() override;
    void execute(const dnnl::stream& strm) override;
    bool created() const override;

    bool needPrepareParams() const override {
        return false;
    };
    void executeDynamicImpl(const dnnl::stream& strm) override;

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

private:
    static std::map<const ov::DiscreteTypeInfo, std::function<void(const std::shared_ptr<ov::Node>&, Math& node)>>&
    getInitializers();

    float alpha = 0.0f;
    float beta = 0.0f;
    float gamma = 0.0f;
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
