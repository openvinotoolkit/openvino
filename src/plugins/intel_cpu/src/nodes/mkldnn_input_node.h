// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <ngraph/op/constant.hpp>
#include <string>

namespace MKLDNNPlugin {

class MKLDNNInputNode : public MKLDNNNode {
public:
    MKLDNNInputNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);
    MKLDNNInputNode(const Shape& shape, const InferenceEngine::Precision &prc, const std::string &name,
                    const std::string &type, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);
    MKLDNNInputNode(MemoryDescPtr memDesc, const std::string &name, const std::string &type, const mkldnn::engine& eng,
                    MKLDNNWeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    bool created() const override;

    void withMeanImage();
    MKLDNNMemoryCPtr getMemoryPtr() const;

    void executeDynamicImpl(mkldnn::stream strm) override {}
    bool isExecutable() const override {
        return false;
    }

    bool needShapeInfer() const override { return false; }
    bool needPrepareParams() const override { return false; }

private:
    void cloneBlobIfRequired();
    void initSupportedPdDefault();
    void initSupportedPdFromMemDesc();

private:
    std::shared_ptr<ngraph::op::Constant> constOp;
    MKLDNNMemoryCPtr memoryPtr;
    MemoryDescPtr extMemDesc = nullptr;
    bool isMeanImage = false;
};

}  // namespace MKLDNNPlugin
