// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common/tile_broadcast_utils.h"

#include <string>

namespace MKLDNNPlugin {

class MKLDNNTileNode : public MKLDNNNode, public TileBroadcastCommon {
public:
    MKLDNNTileNode(const std::shared_ptr<ov::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void execute(mkldnn::stream strm) override;
    void executeDynamicImpl(mkldnn::stream strm) override;
    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

protected:
    bool needPrepareParams() const override;
    void prepareParams() override;
    bool needShapeInfer() const override;
    std::vector<VectorDims> shapeInfer() const override;

private:
    void plainExecute(mkldnn::stream strm);

    static constexpr size_t TILE_INPUT = 0lu;
    static constexpr size_t TILE_REPEATS = 1lu;

    int axis = -1;
    int tiles = 0;
    bool noTiling = false;
    VectorDims originRepeats;

    std::string errorPrefix;
};

}  // namespace MKLDNNPlugin

