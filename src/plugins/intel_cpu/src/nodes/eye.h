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

class EyeLike : public Node {
public:
    EyeLike(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, WeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;
    bool needPrepareParams() const override {return false;};
    void executeDynamicImpl(mkldnn::stream strm) override { execute(strm); }
    std::vector<VectorDims> shapeInfer() const override {
        return {VectorDims{
            // batch_shape
            getRowNum(), getColNum()
        }};
    }

    bool isExecutable() const override;

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

private:
    const std::shared_ptr<ngraph::Node> ngraphOp;
    std::string errorPrefix;
    // do not mismatch
    const size_t getColNum() const {
        return *(reinterpret_cast<size_t *>(getParentEdgeAt(0)->getMemory().GetPtr()));
    }
    const size_t getRowNum() const {
        return (getOriginalInputsNumber() == 1 ?
            getColNum() :
            *(reinterpret_cast<size_t *>(getParentEdgeAt(1)->getMemory().GetPtr())));
    }
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov