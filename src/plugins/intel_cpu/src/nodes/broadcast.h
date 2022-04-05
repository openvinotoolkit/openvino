// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common/tile_broadcast_utils.h"

#include <memory>
#include <string>
#include <vector>

namespace ov {
namespace intel_cpu {
namespace node {

class Broadcast : public Node, public TileBroadcastCommon {
public:
    Broadcast(const std::shared_ptr<ov::Node>& op, const dnnl::engine& eng, WeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void execute(dnnl::stream strm) override;
    void executeDynamicImpl(dnnl::stream strm) override;
    bool created() const override;

    bool isExecutable() const override;
    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

protected:
    bool needPrepareParams() const override;
    void prepareParams() override;
    bool needShapeInfer() const override;
    std::vector<VectorDims> shapeInfer() const override;

private:
    void plainExecute(dnnl::stream strm);

    enum AutoBroadcastType {
        NUMPY,
        EXPLICIT
    };
    AutoBroadcastType broadcastType = NUMPY;

    static constexpr size_t INPUT_DATA_IDX = 0;
    static constexpr size_t TARGET_SHAPE_IDX = 1;
    static constexpr size_t AXES_MAPPING_IDX = 2;

    std::vector<int32_t> targetShape;
    std::vector<int32_t> axesMapping;

    std::string errorPrefix;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
