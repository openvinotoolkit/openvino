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

class ExperimentalDetectronTopKROIs : public Node {
public:
    ExperimentalDetectronTopKROIs(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void execute(const dnnl::stream& strm) override;
    [[nodiscard]] bool created() const override;

    [[nodiscard]] bool needShapeInfer() const override {
        return false;
    };
    [[nodiscard]] bool needPrepareParams() const override {
        return false;
    };
    void executeDynamicImpl(const dnnl::stream& strm) override {
        execute(strm);
    };

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

private:
    // Inputs:
    //      rois, shape [n, 4]
    //      rois_probs, shape [n]
    // Outputs:
    //      top_rois, shape [max_rois, 4]

    const int INPUT_ROIS{0};
    const int INPUT_PROBS{1};

    const int OUTPUT_ROIS{0};
    int max_rois_num_;
};

}  // namespace ov::intel_cpu::node
