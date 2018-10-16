// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>
#include <vector>
#include <memory>

namespace MKLDNNPlugin {

class MKLDNNReshapeNode : public MKLDNNNode {
public:
    MKLDNNReshapeNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng);
    ~MKLDNNReshapeNode() override = default;

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;
    void setDynamicBatchLim(int lim) override;

private:
    static Register<MKLDNNReshapeNode> reg;
    std::shared_ptr<mkldnn::primitive> srcPrim;
    std::shared_ptr<mkldnn::primitive> dstPrim;
    MKLDNNMemoryPtr srcMem;
    MKLDNNMemoryPtr dstMem;

    MKLDNNMemoryPtr dst_blocked;
    MKLDNNMemoryPtr src_blocked;
};

}  // namespace MKLDNNPlugin

