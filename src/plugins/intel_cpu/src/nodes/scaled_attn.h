// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <ie_common.h>
#include <node.h>

#include <memory>
#include <string>
#include <vector>

namespace ov {
namespace intel_cpu {
namespace node {

class ScaledDotProductAttention : public Node {
public:
    ScaledDotProductAttention(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context);

    void getSupportedDescriptors() override {}
    bool created() const override {
        return getType() == Type::ScaledDotProductAttention;
    }
    bool needShapeInfer() const override {
        return false;
    };
    bool needPrepareParams() const override {
        return false;
    };
    bool isExecutable() const override {
        return true;
    }
    void executeDynamicImpl(dnnl::stream strm) override {
        execute(strm);
    }
    void initSupportedPrimitiveDescriptors() override;
    void execute(dnnl::stream strm) override;
    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

    struct Executor {
        virtual void execute(dnnl::stream strm, ScaledDotProductAttention* node) = 0;
    };

    bool is_causal() const {
        return m_is_causal;
    }

    int get_rope_type() const {
        return m_rope_type;
    }

    bool is_out_transpose() const {
        return m_has_out_transpose;
    }

private:
    bool m_is_causal;
    int m_rope_type = -1;  // -1 means no rope
    bool m_has_out_transpose = false;
    std::shared_ptr<Executor> m_executor;
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
