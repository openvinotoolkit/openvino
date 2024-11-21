// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>

namespace ov {
namespace intel_cpu {
namespace node {

class Identity : public Node {
public:

    Identity(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override;

    void initSupportedPrimitiveDescriptors() override;

    bool needPrepareParams() const override;

    void prepareParams() override;

    void execute(dnnl::stream strm) override;

    void executeDynamicImpl(dnnl::stream strm) override;

    bool isExecutable() const override;

    bool created() const override;

    bool canBeInPlace() const override;

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

    std::string getPrimitiveDescriptorType() const override;

protected:
    bool needShapeInfer() const override;

private:
    element::Type m_out_prc;
    bool m_const_input = false;
    VectorDims m_out_shape = {};
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
