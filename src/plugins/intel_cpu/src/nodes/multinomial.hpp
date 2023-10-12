// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <node.h>

#include <string>

namespace ov {
namespace intel_cpu {
namespace node {

class Multinomial : public Node {
public:
    Multinomial(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override; // done
    void initSupportedPrimitiveDescriptors() override; // done

    bool created() const override; // done

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept; // done

    void prepareParams() override; // done

    bool isExecutable() const override; // done
    void execute(dnnl::stream strm) override; // done
    void executeDynamicImpl(dnnl::stream strm) override; // done
    bool canBeInPlace() const override { return false; } // done

private:
    static constexpr size_t PROBS_PORT = 0lu;
    static constexpr size_t NUM_SAMPLES_PORT = 1lu;
    static constexpr size_t OUTPUT_PORT = 0lu;

    ov::element::Type_t m_convert_type = ov::element::i32;
    bool m_with_replacement = false;
    bool m_log_probs = false;
    uint64_t m_global_seed = 0;
    uint64_t m_op_seed = 0;

    std::string m_errorPrefix;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
