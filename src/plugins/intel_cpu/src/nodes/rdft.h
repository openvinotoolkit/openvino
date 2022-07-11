// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <node.h>
#include <string>
#include <map>

namespace ov {
namespace intel_cpu {
namespace node {

struct RDFTExecutor {
    virtual void execute(float* src, float* dst, size_t rank, const std::vector<int>& axes,
                         const VectorDims& input_shape, const VectorDims& output_shape,
                         const VectorDims& input_strides, const VectorDims& output_strides) = 0;
};

class RDFT : public Node {
public:
    RDFT(const std::shared_ptr<ngraph::Node>& op, const dnnl::engine& eng, WeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    bool needPrepareParams() const override;
    void prepareParams() override;
    void execute(dnnl::stream strm) override;
    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

private:
    std::string error_msg_prefix;

    bool inverse;

    std::shared_ptr<RDFTExecutor> executor;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
