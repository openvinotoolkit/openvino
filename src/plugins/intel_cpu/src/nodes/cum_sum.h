// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>

#include "cpu_memory.h"
#include "graph_context.h"
#include "node.h"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu::node {

class CumSum : public Node {
public:
    CumSum(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void execute(const dnnl::stream& strm) override;
    [[nodiscard]] bool created() const override;

    [[nodiscard]] bool needPrepareParams() const override;
    void executeDynamicImpl(const dnnl::stream& strm) override;

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

private:
    template <typename dataType>
    void exec();

    template <bool reverse, bool exclusive, typename dataType>
    void cumSum(const dataType* input, dataType* output, const std::vector<size_t>& strides);

    static void parallelItInit(size_t start, std::vector<size_t>& counters, const std::vector<size_t>& iterationRange);

    static inline void parallelItStep(std::vector<size_t>& counters, const std::vector<size_t>& iterationRange);

    static inline size_t getStartOffset(const std::vector<size_t>& forStartOffset, const std::vector<size_t>& strides);

    [[nodiscard]] size_t getAxis(const IMemory& _axis, const IMemory& _data) const;

    enum : uint8_t { CUM_SUM_DATA, AXIS, numOfInputs };
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

}  // namespace ov::intel_cpu::node
