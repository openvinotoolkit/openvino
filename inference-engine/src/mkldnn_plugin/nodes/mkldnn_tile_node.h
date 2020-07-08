// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>

#include "common/tile_broadcast_utils.h"
#include <ngraph/op/tile.hpp>

namespace MKLDNNPlugin {

class MKLDNNTileNode : public MKLDNNNode, public TileBroadcastCommon {
public:
    MKLDNNTileNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);
    ~MKLDNNTileNode() override = default;

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;

private:
    std::shared_ptr<ngraph::Node> tile;
};

}  // namespace MKLDNNPlugin

