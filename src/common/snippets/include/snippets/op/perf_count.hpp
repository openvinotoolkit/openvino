// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef SNIPPETS_DEBUG_CAPS

#pragma once

#include "openvino/op/op.hpp"
#include "openvino/runtime/threading/thread_local.hpp"

namespace ov {
namespace snippets {

namespace op {
class PerfCountEnd;
} // namespace op

namespace utils {

/**
 * @interface PerfCountDumper
 * @brief Dumper for node debug properties
 * @ingroup snippets
 */
class Dumper {
public:
    Dumper();
    ~Dumper();

    void update(const op::PerfCountEnd* node,
                ov::threading::ThreadLocal<uint64_t> accumulation,
                ov::threading::ThreadLocal<uint32_t> iteration);

private:
    void dump_brgemm_params_to_csv();

    static std::string brgemm_csv_path;
    static std::map<std::string, std::string> m_debug_params_map;
    static size_t nodes_count;
};

} // namespace utils

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
    ov::threading::ThreadLocal<std::chrono::high_resolution_clock::time_point> start_time_stamp;
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
    PerfCountEnd();
    ~PerfCountEnd();

    void output_perf_count();
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override;

    void init_pc_begin();
    void set_accumulated_time();

private:

    ov::threading::ThreadLocal<uint64_t> accumulation;
    ov::threading::ThreadLocal<uint32_t> iteration;

    utils::Dumper csv_dumper;
    std::shared_ptr<PerfCountBegin> m_pc_begin = nullptr;
};

} // namespace op
} // namespace snippets
} // namespace ov
#endif  // SNIPPETS_DEBUG_CAPS
