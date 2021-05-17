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

enum class ScatterUpdateMode {
    ScatterUpdate,
    ScatterNDUpdate,
    ScatterElementsUpdate
};

class MKLDNNScatterUpdateNode : public MKLDNNNode {
public:
    MKLDNNScatterUpdateNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    bool created() const override;
    void execute(mkldnn::stream strm) override;
    bool canBeInPlace() const override {
        return false;
    }

    static bool isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept;

private:
    void scatterUpdate(uint8_t *indicesPtr, uint8_t *updatePtr, int axis, uint8_t *dstDataPtr);
    void scatterNDUpdate(uint8_t *indicesPtr, uint8_t *updatePtr, uint8_t *dstDataPtr);
    void scatterElementsUpdate(uint8_t *indicesPtr, uint8_t *updatePtr, int axis, uint8_t *dstDataPtr);
    inline int64_t getIndicesValue(uint8_t *indices, size_t offset);

    ScatterUpdateMode scatterUpdateMode = ScatterUpdateMode::ScatterUpdate;
    const size_t DATA_ID = 0;
    const size_t INDICES_ID = 1;
    const size_t UPDATE_ID = 2;
    const size_t AXIS_ID = 3;

    // if axis can be set other than default 0.
    bool axisRelaxed = false;
    size_t dataSize, indicesSize, axisSize;
    InferenceEngine::Precision dataPrec, indicesPrec, axisPrec;

    std::string errorPrefix;
};

}  // namespace MKLDNNPlugin