// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include <ie_common.h>
#include <mkldnn_node.h>

namespace MKLDNNPlugin {

class MKLDNNPriorBoxNode : public MKLDNNNode {
public:
    MKLDNNPriorBoxNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;

    bool needShapeInfer() const override;
    std::vector<VectorDims> shapeInfer() const override;
    bool needPrepareParams() const override;

    void executeDynamicImpl(mkldnn::stream strm) override { execute(strm); }

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

private:
    float offset;
    float step;
    std::vector<float> min_size;
    std::vector<float> max_size;
    bool flip;
    bool clip;
    bool scale_all_sizes;

    std::vector<float> fixed_size;
    std::vector<float> fixed_ratio;
    std::vector<float> density;

    std::vector<float> aspect_ratio;
    std::vector<float> variance;

    int number_of_priors;
};

}  // namespace MKLDNNPlugin
