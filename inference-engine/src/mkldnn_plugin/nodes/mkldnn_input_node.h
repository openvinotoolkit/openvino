// Copyright (C) 2018-2021 Intel Corporation
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

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    bool created() const override;

    void withMeanImage();
    MKLDNNMemoryCPtr getMemoryPtr() const;

private:
    void cloneBlobIfRequired();

private:
    std::shared_ptr<ngraph::op::Constant> constOp;
    InferenceEngine::Precision precision;
    MKLDNNMemoryCPtr memoryPtr;
    bool isMeanImage = false;
};

}  // namespace MKLDNNPlugin
