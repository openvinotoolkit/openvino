// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>

namespace MKLDNNPlugin {

class MKLDNNDFTNode : public MKLDNNNode {
public:
    MKLDNNDFTNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);
    ~MKLDNNDFTNode() override = default;

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept;

private:
    void dftNd(float* output, const std::vector<size_t>& outputStrides) const;
    void fft(float* data, int64_t dataLength, bool parallelize = false) const;
    void naiveDFT(float* data, size_t dataLength) const;

    std::vector<std::pair<float, float>> generateTwiddles(size_t n_complex) const;

    std::unordered_map<size_t, std::vector<std::pair<float, float>>> twiddlesMap;
    std::vector<int32_t> axes;
    std::vector<size_t> outputShape;
    std::vector<size_t> inputShape;
    std::string layerErrorPrefix;
    const size_t DATA_INDEX = 0;
    const size_t AXES_INDEX = 1;
    const size_t SIGNAL_SIZE_INDEX = 2;
    static constexpr float PI = 3.141592653589793238462643f;
    bool inverse;
};

}  // namespace MKLDNNPlugin
