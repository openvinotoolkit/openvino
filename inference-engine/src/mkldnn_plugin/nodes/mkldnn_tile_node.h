// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>

namespace MKLDNNPlugin {

class MKLDNNTileNode : public MKLDNNNode {
public:
    MKLDNNTileNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng);
    ~MKLDNNTileNode() override = default;

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;

private:
    static Register<MKLDNNTileNode> reg;
    int axis = 0;
    int tiles = 0;
};

}  // namespace MKLDNNPlugin

