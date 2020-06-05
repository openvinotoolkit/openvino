// Copyright (C) 2020 Intel Corporation
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
    MKLDNNScatterUpdateNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);
    ~MKLDNNScatterUpdateNode() override = default;

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    bool created() const override;
    void execute(mkldnn::stream strm) override;
    bool canBeInPlace() const override {
        return false;
    }

private:
    template <typename data_t, typename index_t>
    void scatterUpdate(data_t *srcData, index_t *indices, data_t *update, int axis, data_t *dstData, ScatterUpdateMode mode);

    ScatterUpdateMode scatterUpdateMode = ScatterUpdateMode::ScatterUpdate;
    const size_t DATA_ID = 0;
    const size_t INDICES_ID = 1;
    const size_t UPDATE_ID = 2;
    int axis = 0;

    mkldnn::primitive_attr attr;
    std::vector<MKLDNNMemoryPtr> PostOpsIntBlobMemory;

    bool planar_layout = true;
    size_t inputSize, indicesSize, outputSize;
    InferenceEngine::Precision inputPrec, indicesPrec, outputPrec;
    size_t blk_size;
};

}  // namespace MKLDNNPlugin