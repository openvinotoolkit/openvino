// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common/tile_broadcast_utils.h"

#include <memory>
#include <string>
#include <vector>


namespace MKLDNNPlugin {

class MKLDNNBroadcastNode : public MKLDNNNode, public TileBroadcastCommon {
public:
    MKLDNNBroadcastNode(const std::shared_ptr<ov::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);
    ~MKLDNNBroadcastNode() override = default;

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(mkldnn::stream strm) override;
    void executeDynamicImpl(mkldnn::stream strm) override {
        execute(strm);
    }
    void plainExecute(mkldnn::stream strm);
    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

protected:
    bool needPrepareParams() const override;
    void prepareParams() override;
    bool needShapeInfer() const override;
    std::vector<VectorDims> shapeInfer() const override;

private:
    enum AutoBroadcastType {
        NUMPY,
        EXPLICIT
    };
    AutoBroadcastType broadcastType;

    static constexpr size_t INPUT_DATA_IDX = 0;
    static constexpr size_t TARGET_SHAPE_IDX = 1;
    static constexpr size_t AXES_MAPPING_IDX = 2;

    // Class members below are mutable due to they are used in constant function shapeInfer.
    // TODO: make shapeInfer not constant?
    mutable std::vector<int32_t> targetShape;
    mutable std::vector<int32_t> axesMapping;
    mutable bool needPrepareParamsVar = false;

    std::string errorPrefix;
};

}  // namespace MKLDNNPlugin
