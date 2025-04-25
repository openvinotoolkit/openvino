// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>

#include <memory>
#include <string>
#include <vector>

namespace ov {
namespace intel_cpu {
namespace node {

class Ngram : public Node {
public:
    Ngram(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override{};
    void initSupportedPrimitiveDescriptors() override;
    void execute(const dnnl::stream& strm) override;
    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

protected:
    void executeDynamicImpl(const dnnl::stream& strm) override;
    void prepareParams() override;

private:
    template <typename idces_type>
    std::vector<size_t> computeBatchLenghts();

    size_t k = 0;
    size_t windowSize = 0;
    size_t windowStride = 0;

    size_t leftPad = 0;
    size_t rightPad = 0;
    size_t leftPaddingSize = 0;
    size_t rightPaddingSize = 0;

    size_t idcesShapeSize = 0;
    size_t idcesStride = 0;
    size_t numIdces = 0;
    size_t numOutElems = 0;

    ov::element::Type idcesPrecision;
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
