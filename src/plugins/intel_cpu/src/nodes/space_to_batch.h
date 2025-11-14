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

class SpaceToBatch : public Node {
public:
    SpaceToBatch(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void execute(const dnnl::stream& strm) override;
    bool created() const override;

    bool needPrepareParams() const override {
        return false;
    };
    bool needShapeInfer() const override {
        return true;
    };
    void executeDynamicImpl(const dnnl::stream& strm) override;

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

private:
    std::vector<size_t> blockShapeIn;
    std::vector<size_t> padsBeginIn;

    template <typename T>
    void SpaceToBatchKernel();
};

}  // namespace ov::intel_cpu::node
