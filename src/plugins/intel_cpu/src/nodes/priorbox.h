// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "node.h"

namespace ov {
namespace intel_cpu {
namespace node {

class PriorBox : public Node {
public:
    PriorBox(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override{};
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(const dnnl::stream& strm) override;
    bool created() const override;

    bool needShapeInfer() const override;
    bool needPrepareParams() const override;

    void executeDynamicImpl(const dnnl::stream& strm) override {
        execute(strm);
    }

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

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

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
