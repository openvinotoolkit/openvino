// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>

#include "graph_context.h"
#include "node.h"
#include "openvino/core/node.hpp"

namespace ov::intel_cpu::node {

class CTCLoss : public Node {
public:
    CTCLoss(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void execute(const dnnl::stream& strm) override;
    [[nodiscard]] bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

    void executeDynamicImpl(const dnnl::stream& strm) override;
    [[nodiscard]] bool needPrepareParams() const override {
        return false;
    };

private:
    bool ctcMergeRepeated;
    bool preprocessCollapseRepeated;
    bool unique;
};

}  // namespace ov::intel_cpu::node
