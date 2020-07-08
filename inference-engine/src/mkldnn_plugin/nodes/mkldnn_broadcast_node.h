// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>

#include "common/tile_broadcast_utils.h"
#include <ngraph/op/broadcast.hpp>

namespace MKLDNNPlugin {

class MKLDNNBroadcastNode : public MKLDNNNode, public TileBroadcastCommon {
public:
    MKLDNNBroadcastNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);
    ~MKLDNNBroadcastNode() override = default;

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;

private:
    ngraph::op::AutoBroadcastType broadcastType = ngraph::op::AutoBroadcastType::NUMPY;
    std::vector<size_t> axesMapping;

    std::shared_ptr<ngraph::Node> broadcast;
};

}  // namespace MKLDNNPlugin
