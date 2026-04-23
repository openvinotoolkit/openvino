// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>

#include "cpu_types.h"
#include "graph_context.h"
#include "node.h"
#include "openvino/core/node.hpp"

namespace ov::intel_cpu::node {

class PagedCausalConv1D : public Node {
public:
    PagedCausalConv1D(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

    void getSupportedDescriptors() override {}
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override {}
    void execute(const dnnl::stream& strm) override;
    void executeDynamicImpl(const dnnl::stream& strm) override {
        execute(strm);
    }

    bool created() const override {
        return getType() == Type::PagedCausalConv1D;
    }

    // conv_bias (port 3) may have zero dimension when bias is absent
    bool neverExecute() const override {
        return getSelectedPrimitiveDescriptor()->hasZeroInputDimsAtPort(0);
    }

    // conv_bias (port 3) may have zero dimension when bias is absent
    bool isExecutable() const override {
        return !isInputTensorAtPortEmpty(0);
    }

    bool needPrepareParams() const override {
        return false;
    }
};

}  // namespace ov::intel_cpu::node
