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

class BatchToSpace : public Node {
public:
    BatchToSpace(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;

    [[nodiscard]] bool neverExecute() const override {
        const auto& spd = getSelectedPrimitiveDescriptor();
        return spd->hasZeroInputDims() || spd->hasZeroOutputDims();
    }

    // output shape can potentially be empty
    [[nodiscard]] bool isExecutable() const override {
        return !hasEmptyInputTensors() && !hasEmptyOutputTensors();
    }

    void execute(const dnnl::stream& strm) override;
    [[nodiscard]] bool created() const override;

    [[nodiscard]] bool needPrepareParams() const override {
        return false;
    };
    [[nodiscard]] bool needShapeInfer() const override {
        return true;
    };
    void executeDynamicImpl(const dnnl::stream& strm) override;

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

private:
    template <typename T>
    void batchToSpaceKernel();

    std::vector<size_t> blockShapeIn;
    std::vector<size_t> cropsBeginIn;
};

}  // namespace ov::intel_cpu::node
