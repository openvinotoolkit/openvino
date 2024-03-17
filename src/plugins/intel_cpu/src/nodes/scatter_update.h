// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "node.h"
#include "openvino/op/scatter_elements_update.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

using Reduction = ov::op::v12::ScatterElementsUpdate::Reduction;

enum class ScatterUpdateMode {
    ScatterUpdate,
    ScatterNDUpdate,
    ScatterElementsUpdate
};

class ScatterUpdate : public Node {
public:
    ScatterUpdate(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    bool created() const override;
    void execute(dnnl::stream strm) override;
    bool canBeInPlace() const override {
        return false;
    }

    bool needPrepareParams() const override;
    void executeDynamicImpl(dnnl::stream strm) override;

    bool isExecutable() const override;
    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

    struct Config {
        Reduction reduction_type;
        bool use_init_val = true;
    };

private:
    void scatterUpdate(uint8_t *indicesPtr, uint8_t *updatePtr, int axis, uint8_t *dstDataPtr);
    void scatterNDUpdate(uint8_t *indicesPtr, uint8_t *updatePtr, uint8_t *dstDataPtr);

    inline int64_t getIndicesValue(uint8_t *indices, size_t offset);

    ScatterUpdateMode scatterUpdateMode = ScatterUpdateMode::ScatterUpdate;
    enum { DATA_ID, INDICES_ID, UPDATE_ID, AXIS_ID };

    Config m_config;

    // if axis can be set other than default 0.
    bool axisRelaxed = false;
    size_t dataSize, indicesSize, axisSize;
    ov::element::Type dataPrec, indicesPrec, axisPrec;

    std::string errorPrefix;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
