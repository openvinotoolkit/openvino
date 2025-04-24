// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/emitter.hpp"

#include "openvino/op/op.hpp"

namespace ov {
namespace snippets {
namespace op {

/**
 * @interface LoopBase
 * @brief Base class for LoopBegin and LoopEnd
 * @ingroup snippets
 */
class LoopBase : public ov::op::Op {
public:
    OPENVINO_OP("LoopBase", "SnippetsOpset");
    LoopBase(const std::vector<Output<Node>>& args);
    LoopBase() = default;
protected:
};
class LoopEnd;
/**
 * @interface LoopBegin
 * @brief Marks the start of the Loop region.
 *        Number of outputs always equals to the number of inputs (bypassed values) + 1 (edge to the corresponding LoopEnd)
 * @param args - vector of input values, they are passed directly to output.
 * @ingroup snippets
 */
class LoopBegin : public LoopBase {
    friend LoopEnd;

public:
    OPENVINO_OP("LoopBegin", "SnippetsOpset", LoopBase);
    LoopBegin();

    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override;
    std::shared_ptr<LoopEnd> get_loop_end() const;

protected:
    void validate_and_infer_types_except_LoopEnd();
};

/**
 * @interface LoopEnd
 * @brief Marks the end of the Loop region and defines the loop properties.
 *        Number of outputs always equals to the number of inputs (bypassed values) - 1 (edge to the corresponding LoopEnd)
 * @param args vector of input values + LoopBegin, all values except for the LoopBegin are passed directly to output.
 * @param work_amount total number of evaluations to be processed by the loop
 * @param increment number of evaluations processed in one iteration of the loop.
 * @param is_incremented describes which data pointers attributed to the loop should be incremented on every iteration.
 * @param ptr_increments specifies i/o pointer increment performed on every iteration if the following is_incremented[i] is true
 * @param finalization_offsets pointer increments that are be applied to i/o pointers before exiting the loop
 * @param id the identifier of Loop in Loop system in LoopManager
 * @ingroup snippets
 */
class LoopEnd : public LoopBase {
public:
    OPENVINO_OP("LoopEnd", "SnippetsOpset", LoopBase);
    LoopEnd() = default;
    LoopEnd(const Output<Node>& loop_begin, size_t work_amount, size_t work_amount_increment,
            std::vector<bool> is_incremented, std::vector<int64_t> ptr_increments, std::vector<int64_t> finalization_offsets,
            std::vector<int64_t> element_type_sizes, size_t input_num, size_t output_num, size_t id);

    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override;

    std::shared_ptr<LoopBegin> get_loop_begin();
    const std::vector<bool>& get_is_incremented() const;
    const std::vector<int64_t>& get_finalization_offsets() const;
    const std::vector<int64_t>& get_ptr_increments() const;
    const std::vector<int64_t>& get_element_type_sizes() const;
    size_t get_work_amount() const;
    size_t get_increment() const;
    size_t get_id() const;
    size_t get_input_num() const;
    size_t get_output_num() const;
    bool get_evaluate_once() const;
    bool has_dynamic_params() const;

    void set_is_incremented(std::vector<bool> is_incremented);
    void set_finalization_offsets(std::vector<int64_t> offsets);
    void set_ptr_increments(std::vector<int64_t> new_ptr_increments);
    void set_work_amount(size_t new_work_amount);
    void set_increment(size_t new_increment);
    void set_evaluate_once(bool once);
    void set_id(size_t id);

protected:
    std::vector<bool> m_is_incremented = {};
    std::vector<int64_t> m_ptr_increments = {};
    std::vector<int64_t> m_finalization_offsets = {};
    std::vector<int64_t> m_element_type_sizes = {};
    size_t m_work_amount = 0;
    size_t m_work_amount_increment = 0;
    size_t m_input_num = 0;
    size_t m_output_num = 0;
    size_t m_id = 0;  // the corresponding Loop identificator in LoopManager

    bool m_evaluate_once = false; // true if the Loop is executed only once, used to skip setting and testing the loop counter
};

} // namespace op
} // namespace snippets
} // namespace ov
