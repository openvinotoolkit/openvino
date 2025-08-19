// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>

#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>

#include "graph_context.h"
#include "openvino/core/node.hpp"

namespace ov::intel_cpu::node {

class ShapeOf : public Node {
public:
    ShapeOf(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void initOptimalPrimitiveDescriptor() override;
    void execute(const dnnl::stream& strm) override;
    bool created() const override;
    bool needPrepareParams() const override {
        return false;
    };

    void executeDynamicImpl(const dnnl::stream& strm) override {
        execute(strm);
    }

    bool neverExecute() const override {
        return false;
    };
    bool isExecutable() const override {
        return true;
    }

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;
};

}  // namespace ov::intel_cpu::node
