// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <node.h>
#include "embedding_bag_sum.h"
#include <string>
#include <memory>
#include <vector>

namespace ov {
namespace intel_cpu {
namespace node {

class EmbeddingBagPackedSum : public Node, public EmbeddingBagSum {
public:
    EmbeddingBagPackedSum(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void execute(dnnl::stream strm) override;
    bool created() const override;

    bool isExecutable() const override;
    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

protected:
    void prepareParams() override;
    void executeDynamicImpl(dnnl::stream strm) override;

private:
    void initFromInputs() override;
    void getIndices(size_t embIndex, const int*& indices, size_t& size, int& weightsIdx, bool& withWeight) override;

    const int* _indices = nullptr;
    size_t _batch = 0;
    size_t _indicesPerBag = 0;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
