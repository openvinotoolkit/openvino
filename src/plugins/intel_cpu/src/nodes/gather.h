// Copyright (C) 2018-2024 Intel Corporation
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

class Gather : public Node {
public:
    Gather(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(dnnl::stream strm) override;
    bool created() const override;
    bool isExecutable() const override;
    void resolveInPlaceEdges(Edge::LOOK look) override;

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

    struct threadExecParams {
        std::vector<int> specIdxInBytes;
        std::vector<int> permIdxMask;
        std::vector<int> srcBeforeAxisDiff;
        std::vector<int> idxBatchSumInBytes;
        std::vector<int> dataBeforeAxisSumInBytes;

        std::vector<int> afterAxIdxInBytes;
        std::vector<int> specIdxDiff;
        std::vector<int> beforeAxPermMask;
        std::vector<int> afterAxPermMask;
        int betweenBatchAndAxisIter = 0;
        int specIdxAndAfterAxIterB = 0;

        uint64_t workAmount = 0;
        uint64_t dstStart = 0;
    };

    template <typename OUT_TYPE, typename IN_TYPE>
    void execCompressed8Bit();
    static int8_t get_i4(const uint8_t& val, bool high);
    static int8_t get_u4(const uint8_t& val, bool high);
    template <typename OUT_TYPE, int8_t get4Bit(const uint8_t&, bool)>
    void execCompressed4Bit();

protected:
    void executeDynamicImpl(dnnl::stream strm) override;
    bool needPrepareParams() const override;
    void prepareParams() override;

private:
    void initShortParams(threadExecParams& p, uint64_t start);
    void execReference();

    bool canOptimize1DCase = false;
    void exec1DCase();

    bool compressed = false;
    void execCompressed();

    bool isDataShapeStat = false;
    bool isIdxShapeStat = false;
    bool isAxisInputConst = false;

    bool reverseIndexing = false;

    uint64_t dataTypeSize = 1lu;
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
    uint64_t afterAxisSizeInBytes = 0lu;
    uint64_t axisAndAfterAxisSizeInBytes = 0lu;
    uint64_t axisAndAfterAxisSize = 0lu;
    uint64_t srcAfterBatchSizeInBytes = 0lu;
    uint64_t srcAfterBatchSize = 0lu;
    uint64_t specIdxAndAfterAxSizeB = 0lu;
    uint64_t specIdxAndAfterAxSize = 0lu;
    uint64_t totalWork = 0lu;

    std::vector<threadExecParams> execParamsPerThread;
    std::vector<int> constIndices;

    static constexpr size_t GATHER_DATA = 0;
    static constexpr size_t GATHER_INDICES = 1;
    static constexpr size_t GATHER_AXIS = 2;
    static constexpr size_t GATHER_SCALE = 3;
    static constexpr size_t GATHER_ZP = 4;

    bool have_zp = false;
    bool have_scalar_zp = false;
    bool have_scalar_scale = false;
    size_t zp_group_size = 1u;
    size_t scale_group_size = 1u;
    size_t m_threads_num = 0lu;

    std::shared_ptr<jitGatherKernelBase> jitKernel;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
