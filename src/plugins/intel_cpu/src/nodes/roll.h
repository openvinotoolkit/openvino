// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <node.h>
#include <string>

namespace ov {
namespace intel_cpu {
namespace node {

class Roll : public Node {
public:
    Roll(const std::shared_ptr<ngraph::Node>& op, const dnnl::engine& eng, WeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(dnnl::stream strm) override;
    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

private:
    size_t calculateShiftOffset(size_t dataOffset, size_t dimShift, size_t segmentSize, size_t dimSize);

    template <typename DataType>
    void rollImpl();

    std::vector<size_t> shape;
    static const std::vector<size_t> supportedPrecisionSizes;
    std::string layerErrorPrefix;
    size_t numOfDims;

    const size_t DATA_INDEX = 0ul;
    const size_t SHIFT_INDEX = 1ul;
    const size_t AXES_INDEX = 2ul;
    const size_t numberOfInputs = 3ul;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
