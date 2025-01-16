// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "embedding_bag.h"
#include "node.h"

namespace ov {
namespace intel_cpu {
namespace node {

class EmbeddingBagPacked : public Node, public EmbeddingBag {
public:
    EmbeddingBagPacked(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void execute(dnnl::stream strm) override;
    bool created() const override;

    bool isExecutable() const override;
    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

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
