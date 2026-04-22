// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>

#include "config.h"
#include "cpu_types.h"
#include "graph_context.h"
#include "node.h"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu::node {

class GatedDeltaNet : public Node {
public:
    GatedDeltaNet(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override {}
    bool created() const override {
        return getType() == Type::GatedDeltaNet;
    }

    bool isExecutable() const override {
        return true;
    }

    bool needPrepareParams() const override {
        return false;
    }

    void createPrimitive() override;

    void executeDynamicImpl(const dnnl::stream& strm) override {
        execute(strm);
    }
    void initSupportedPrimitiveDescriptors() override;
    void execute(const dnnl::stream& strm) override;
    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

private:
    MemoryPtr m_tmpInpBuffer = nullptr;
    bool m_fuse_qk_l2norm = false;
    float m_q_l2_norm_eps = 1e-6F;
    float m_k_l2_norm_eps = 1e-6F;
};

}  // namespace ov::intel_cpu::node
