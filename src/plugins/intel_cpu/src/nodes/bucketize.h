// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "node.h"

namespace ov {
namespace intel_cpu {
namespace node {

class Bucketize : public Node {
public:
    Bucketize(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override{};
    void initSupportedPrimitiveDescriptors() override;
    void execute(const dnnl::stream& strm) override;
    bool created() const override;
    void executeDynamicImpl(const dnnl::stream& strm) override {
        execute(strm);
    }

    void prepareParams() override;

    bool neverExecute() const override;
    bool isExecutable() const override;
    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

private:
    template <typename T, typename T_BOUNDARIES, typename T_IND>
    void bucketize();

    const size_t INPUT_TENSOR_PORT = 0;
    const size_t INPUT_BINS_PORT = 1;
    const size_t OUTPUT_TENSOR_PORT = 0;

    size_t num_values = 0;
    size_t num_bin_values = 0;
    bool with_right = false;
    bool with_bins = false;

    ov::element::Type input_precision;
    ov::element::Type boundaries_precision;
    ov::element::Type output_precision;
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
