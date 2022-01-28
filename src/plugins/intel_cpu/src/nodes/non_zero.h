// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>
#include <string>
#include <memory>
#include <vector>
#include <dnnl_extension_utils.h>

namespace ov {
namespace intel_cpu {
namespace node {

class NonZero : public Node {
public:
  NonZero(const std::shared_ptr<ngraph::Node>& op, const dnnl::engine& eng, WeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void execute(dnnl::stream strm) override;
    bool created() const override;
    bool needShapeInfer() const override {return false;};
    bool needPrepareParams() const override {return false;};
    void executeDynamicImpl(dnnl::stream strm) override;
    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

    bool isExecutable() const override { return true; }

private:
    std::string errorPrefix;
    template <typename inputType>
    void executeSpecified();
    template<typename T>
    struct NonZeroExecute;
    template <typename T>
    std::vector<size_t> getNonZeroElementsCount(const T* arg, const Shape& arg_shape);
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
