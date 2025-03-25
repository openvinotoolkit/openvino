// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "node.h"

namespace ov {
namespace intel_cpu {
namespace node {

class CumSum : public Node {
public:
    CumSum(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override{};
    void initSupportedPrimitiveDescriptors() override;
    void execute(const dnnl::stream& strm) override;
    bool created() const override;

    bool needPrepareParams() const override;
    void executeDynamicImpl(const dnnl::stream& strm) override;

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

private:
    template <typename dataType>
    void exec();

    template <bool reverse, bool exclusive, typename dataType>
    void cumSum(const dataType* input, dataType* output, const std::vector<size_t>& strides);

    void parallelItInit(size_t start, std::vector<size_t>& counters, const std::vector<size_t>& iterationRange);

    inline void parallelItStep(std::vector<size_t>& counters, const std::vector<size_t>& iterationRange);

    inline size_t getStartOffset(const std::vector<size_t>& forStartOffset, const std::vector<size_t>& strides) const;

    size_t getAxis(const IMemory& _axis, const IMemory& _data) const;

    enum { CUM_SUM_DATA, AXIS, numOfInputs };
    bool exclusive;
    bool reverse;
    size_t numOfDims;
    size_t axis = 0;

    ov::element::Type dataPrecision;

    template <typename T>
    struct CumSumExecute {
        void operator()(CumSum* node) {
            node->exec<T>();
        }
    };
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
