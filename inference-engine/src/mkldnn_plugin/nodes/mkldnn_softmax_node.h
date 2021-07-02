// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>
#include <memory>
#include <vector>

namespace MKLDNNPlugin {

class MKLDNNSoftMaxNode : public MKLDNNNode {
public:
    MKLDNNSoftMaxNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    void initOptimalPrimitiveDescriptor() override;
    void createDescriptor(const std::vector<const MemoryDesc*>& inputDesc,
                          const std::vector<const MemoryDesc*>& outputDesc) override;
    void getSupportedDescriptors() override;
    void createPrimitive() override;
    bool created() const override;

private:
    size_t axis = 0;
};

}  // namespace MKLDNNPlugin

