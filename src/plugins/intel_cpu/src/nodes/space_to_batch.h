// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <node.h>
#include <string>
#include <memory>
#include <vector>

namespace ov {
namespace intel_cpu {
namespace node {

class SpaceToBatch : public Node {
public:
    SpaceToBatch(const std::shared_ptr<ngraph::Node>& op, const dnnl::engine& eng, WeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void execute(dnnl::stream strm) override;
    bool created() const override;

    std::vector<VectorDims> shapeInfer() const override;
    bool needPrepareParams() const override { return false; };
    void executeDynamicImpl(dnnl::stream strm) override;

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

private:
    std::vector<size_t> blockShapeIn;
    std::vector<size_t> padsBeginIn;

    std::string errorPrefix;

    template<typename T>
    void SpaceToBatchKernel();
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
