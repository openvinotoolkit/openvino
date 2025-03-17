// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef SNIPPETS_DEBUG_CAPS

#pragma once

#include <chrono>

#include "openvino/op/op.hpp"
#include "openvino/runtime/threading/thread_local.hpp"

namespace ov {
namespace snippets {

namespace op {
class PerfCountEnd;
} // namespace op

namespace utils {

/**
 * @interface Dumper
 * @brief Dumper for node debug properties
 * @ingroup snippets
 */
class Dumper {
public:
    Dumper() = default;
    virtual ~Dumper() = default;

    void init(const std::string &params);
    virtual void update(const op::PerfCountEnd* node) = 0;
protected:
    std::map<std::string, std::string> m_debug_params_map;
    std::string m_params;
};

/**
 * @interface ConsoleDumper
 * @brief Dumper for node debug properties (output: stdout)
 * @ingroup snippets
 */
class ConsoleDumper : public Dumper {
public:
    ConsoleDumper() = default;
    ~ConsoleDumper() override;

    void update(const op::PerfCountEnd* node) override;

private:
    ov::threading::ThreadLocal<uint64_t> m_accumulation;
    ov::threading::ThreadLocal<uint32_t> m_iteration;
};

/**
 * @interface CSVDumper
 * @brief Dumper for node debug properties (output: .csv file)
 * @ingroup snippets
 */
class CSVDumper : public Dumper {
public:
    CSVDumper(const std::string csv_path);
    ~CSVDumper() override;

    void update(const op::PerfCountEnd* node) override;

private:
    const std::string csv_path;
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
    PerfCountEnd(const Output<Node>& pc_begin,
                 std::vector<std::shared_ptr<utils::Dumper>> dumpers = {},
                 const std::string& params = "");
    PerfCountEnd();
    ~PerfCountEnd();

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override;

    void init_pc_begin();
    void set_accumulated_time();

    const ov::threading::ThreadLocal<uint64_t> &get_accumulation() const {
        return accumulation;
    }

    const ov::threading::ThreadLocal<uint32_t> &get_iteration() const {
        return iteration;
    }

private:
    ov::threading::ThreadLocal<uint64_t> accumulation;
    ov::threading::ThreadLocal<uint32_t> iteration;

    std::vector<std::shared_ptr<utils::Dumper>> dumpers;
    std::shared_ptr<PerfCountBegin> m_pc_begin = nullptr;
};

} // namespace op
} // namespace snippets
} // namespace ov
#endif  // SNIPPETS_DEBUG_CAPS
