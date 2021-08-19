// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>

namespace MKLDNNPlugin {

class MKLDNNRollNode : public MKLDNNNode {
public:
    MKLDNNRollNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept;

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

}  // namespace MKLDNNPlugin
