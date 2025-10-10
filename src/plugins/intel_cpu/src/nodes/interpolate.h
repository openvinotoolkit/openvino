// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cassert>
#include <common/primitive_attr.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>

#include "cpu_types.h"
#include "executors/interpolate_config.hpp"
#include "graph_context.h"
#include "node.h"
#include "nodes/executors/executor_factory.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu::node {

class Interpolate : public Node {
public:
    static constexpr size_t DATA_ID = 0;
    static constexpr size_t TARGET_SHAPE_ID = 1;
    static constexpr size_t SCALES_ID = 2;
    static constexpr size_t AXES_ID = 3;
    static constexpr size_t SIZE_OR_SCALE_ID_V11 = 1;
    static constexpr size_t AXES_ID_V11 = 2;
    static constexpr int CUBIC_GRID_LEN = 4;
    static constexpr float PILLOW_BILINEAR_WINDOW_SCALE = 1.0F;
    static constexpr float PILLOW_BICUBIC_WINDOW_SCALE = 2.0F;

    Interpolate(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    bool created() const override;
    void execute(const dnnl::stream& strm) override;
    void executeDynamicImpl(const dnnl::stream& strm) override;
    bool canBeInPlace() const override {
        return false;
    }
    bool canFuse(const NodePtr& node) const override;

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

    bool needShapeInfer() const override;
    bool needPrepareParams() const override;
    void prepareParams() override;

    inline int get_scale_id() const;
    inline int get_axis_id() const;

private:
    bool is_version11 = true;
    InterpolateAttrs interpAttrs;
    size_t dataRank = 0;

    // New executor pipeline
    ExecutorFactoryPtr<InterpolateAttrs> factory;
    ExecutorPtr executor = nullptr;
    MemoryArgs memory;
    std::unordered_map<int, int> m_atoi;

    // legacy path removed

    static VectorDims getPaddedInputShape(const VectorDims& srcDims,
                                          const std::vector<int>& padBegin,
                                          const std::vector<int>& padEnd);
    std::vector<float> getScales(const VectorDims& srcDimPad, const VectorDims& dstDim);
    static size_t getSpatialDimsNum(const std::vector<float>& scales);

    bool hasPad = false;

    bool isAxesSpecified = false;
    std::vector<int> axes;
    std::vector<float> scales;
    bool isScaleConstant = false;

    // legacy buffers removed

    std::vector<float> lastScales;
    std::vector<int32_t> lastSizes;

    VectorDims lastOutputDims;

    // acl path is handled in implementations list
};

}  // namespace ov::intel_cpu::node
