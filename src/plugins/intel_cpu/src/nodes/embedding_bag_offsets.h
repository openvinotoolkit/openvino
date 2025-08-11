// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>

#include "embedding_bag.h"
#include "graph_context.h"
#include "node.h"
#include "openvino/core/node.hpp"

namespace ov::intel_cpu::node {

class EmbeddingBagOffset : public Node, public EmbeddingBag {
public:
    EmbeddingBagOffset(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void execute(const dnnl::stream& strm) override;
    bool created() const override;

    bool neverExecute() const override;
    bool isExecutable() const override;
    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

protected:
    void prepareParams() override;
    void executeDynamicImpl(const dnnl::stream& strm) override;

private:
    void initFromInputs() override;
    void getIndices(size_t embIndex, const int*& indices, size_t& size, int& weightsIdx, bool& withWeight) override;

    const size_t OFFSETS_IDX = 2LU;

    const int* indicesData_ = nullptr;
    const int* offsetsData_ = nullptr;
    const int* defaultIndices_ = nullptr;

    size_t _indicesLen = 0;
    size_t _offsetsLen = 0;
};

}  // namespace ov::intel_cpu::node
