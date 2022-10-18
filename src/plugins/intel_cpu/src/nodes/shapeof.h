// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>
#include <string>
#include <memory>
#include <vector>
#include <dnnl_extension_utils.h>

namespace ov {
namespace intel_cpu {
namespace node {

class ShapeOf : public Node {
public:
    ShapeOf(const std::shared_ptr<ngraph::Node>& op, const dnnl::engine& eng, WeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void execute(dnnl::stream strm) override;
    bool created() const override;
    bool needPrepareParams() const override {return false;};
    void executeDynamicImpl(dnnl::stream strm) override { execute(strm); }
    std::vector<VectorDims> shapeInfer() const override {
        return {VectorDims{getParentEdgesAtPort(0)[0]->getMemory().getStaticDims().size()}};
    }

    bool isExecutable() const override;

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

private:
    std::string errorPrefix;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
