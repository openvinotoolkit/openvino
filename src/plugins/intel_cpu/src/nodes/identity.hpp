// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>

namespace ov::intel_cpu::node {

class Identity : public Node {
public:
    Identity(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override;

    void initSupportedPrimitiveDescriptors() override;

    void prepareParams() override;

    void execute(const dnnl::stream& strm) override;

    void executeDynamicImpl(const dnnl::stream& strm) override;

    bool isExecutable() const override;

    bool created() const override;

    bool canBeInPlace() const override;

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

protected:
    bool needShapeInfer() const override;

private:
    element::Type m_out_prc;
    bool m_const_input = false;
    VectorDims m_out_shape;
};

}  // namespace ov::intel_cpu::node
