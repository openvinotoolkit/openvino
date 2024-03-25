// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>
#include "kernels/x64/gather_uni_kernel.hpp"

#include <memory>
#include <string>
#include <vector>

namespace ov {
namespace intel_cpu {
namespace node {

class GatherCompressed : public Node {
public:
    GatherCompressed(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void execute(dnnl::stream strm) override;
    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

protected:
    void executeDynamicImpl(dnnl::stream strm) override;
    bool needPrepareParams() const override;
    void prepareParams() override;

private:
    void execReference();

    template <typename IN_TYPE, typename OUT_TYPE>
    void execReference8bit();

    template <typename OUT_TYPE>
    void execReferenceU4();
    template <typename OUT_TYPE>
    void execReferenceI4();

    bool isDataShapeStat = false;
    bool isIdxShapeStat = false;
    bool isAxisInputConst = false;

    bool reverseIndexing = true;

    static constexpr uint64_t idxTypeSize = sizeof(int);

    int axis = 0;
    int axisDim = 0;
    int batchDims = 0;
    int dataSrcRank = 1;
    uint64_t specIndicesSize = 0lu;
    uint64_t beforeBatchSize = 0lu;
    uint64_t beforeAxisSize = 0lu;
    uint64_t betweenBatchAndAxisSize = 0lu;
    uint64_t afterAxisSize = 0lu;
    uint64_t axisAndAfterAxisSize = 0lu;
    uint64_t srcAfterBatchSize = 0lu;
    uint64_t specIdxAndAfterAxSizeB = 0lu;
    uint64_t totalWork = 0lu;

    static constexpr size_t GATHER_DATA = 0;
    static constexpr size_t GATHER_INDICES = 1;
    static constexpr size_t GATHER_AXIS = 2;
    static constexpr size_t GATHER_SCALE = 3;
    static constexpr size_t GATHER_ZP = 4;

    bool have_zp = false;
    size_t zp_group_size = 1u;
    size_t scale_group_size = 1u;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov