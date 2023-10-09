// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace snippets {
namespace op {

/**
 * @interface PerfCountBeginBase
 * @brief Base class for PerfCountBegin and PerfCountRdtscBegin(cpu)
 * @ingroup snippets
 */
class PerfCountBeginBase : public ov::op::Op {
public:
    OPENVINO_OP("PerfCountBeginBase", "SnippetsOpset");
    PerfCountBeginBase(const std::vector<Output<Node>>& args);
    PerfCountBeginBase() = default;

    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;

protected:
    void validate_and_infer_types_except_PerfCountEnd();
};

/**
 * @interface PerfCountEndBase
 * @brief Base class for PerfCountEnd and PerfCountRdtscEnd
 * @ingroup snippets
 */
class PerfCountEndBase : public ov::op::Op {
public:
    OPENVINO_OP("PerfCountEndBase", "SnippetsOpset");
    PerfCountEndBase(const std::vector<Output<Node>>& args);
    PerfCountEndBase() = default;

    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;
};

/**
 * @interface PerfCountBegin
 * @brief Performance count start time with chrono call
 * @ingroup snippets
 */
class PerfCountBegin : public PerfCountBeginBase {
public:
    OPENVINO_OP("PerfCountBegin", "SnippetsOpset", PerfCountBeginBase);
    PerfCountBegin();

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override;

    void set_start_time();
    std::chrono::high_resolution_clock::time_point& get_start_time();

private:
    std::chrono::high_resolution_clock::time_point start_time_stamp = {};
};

/**
 * @interface PerfCountEnd
 * @brief Performance count end time and duration with chrono call
 * @ingroup snippets
 */
class PerfCountEnd : public PerfCountEndBase {
public:
    OPENVINO_OP("PerfCountEnd", "SnippetsOpset", PerfCountEndBase);
    PerfCountEnd(const Output<Node>& pc_begin);
    PerfCountEnd() = default;
    ~PerfCountEnd() {
        uint64_t avg = iteration == 0 ? 0 : accumulation / iteration;
        std::cout << "accumulation:" << accumulation << "ns, iteration:" << iteration << " avg:" << avg << "ns"<< std::endl;
    }
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override;

    std::shared_ptr<PerfCountBegin> get_pc_begin();
    void set_accumulated_time();

private:
    uint64_t accumulation = 0ul;
    uint32_t iteration = 0u;
};

} // namespace op
} // namespace snippets
} // namespace ov
