// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"
#include "snippets/emitter.hpp"
#include "ngraph/op/parameter.hpp"

namespace ngraph {
namespace snippets {
namespace op {

/**
 * @interface LoopBase
 * @brief Inserted during scheduling generation and represents Loop in affine notation
 * @ingroup snippets
 */
class LoopBase : public ngraph::op::Op {
public:
    OPENVINO_OP("LoopBase", "SnippetsOpset");
    LoopBase(const std::vector<Output<Node>>& args, size_t dimension, size_t work_amount, size_t increment);
    LoopBase() = delete;
    bool visit_attributes(AttributeVisitor& visitor) override;
    size_t get_work_amount() const;
    size_t get_increment() const;
    size_t get_dimension() const;
    bool get_evaluate_once() const;

protected:
    size_t dimension;
    size_t work_amount;
    size_t wa_increment;
    bool evaluate_once; // true if the Loop is executed only once, used to skip setting and testing the loop counter
};
class LoopEnd;
class LoopBegin : public LoopBase {
    friend LoopEnd;

public:
    OPENVINO_OP("LoopBegin", "SnippetsOpset");
    /// \brief Construct an Loop
    /// \param region The vector of pairs: emitters and the corresponding registers
    /// \param increment Loop size - count of elements to load and store.
    ///                  Vector Loop should have size of vector register and Scalar Loop should have 1
    /// \param num_inputs Count of inputs
    /// \param num_outputs Count of outputs
    /// \param io_dims Vector of last dimensions of inputs and outputs
    /// \param io_data_sizes Vector of data type sizes of inputs and outputs
    explicit LoopBegin(const std::vector<Output<Node>>& args);
    LoopBegin() = delete;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs)  const override;
    std::shared_ptr<LoopEnd> get_loop_end();
    // begin_address and input_regs are needed to communicate information between LoopBegin and LoopEnd emitters
    const uint8_t* begin_address;
    std::vector<size_t> input_regs;
    // true if scalar loop is not needed for tile processing;
    bool avoid_scalar_loop_injection;

private:
    void validate_and_infer_types_except_LoopEnd();
    LoopBegin(const std::vector<Output<Node>>& args, size_t dimension, size_t work_amount, size_t wa_increment);
};

class LoopEnd : public LoopBase {
public:
    OPENVINO_OP("LoopEnd", "SnippetsOpset");
    LoopEnd(const std::vector<Output<Node>>& args, size_t dimension, size_t work_amount, size_t wa_increment,
              std::vector<bool> apply_increment, std::vector<int64_t> finalization_offsets);
    LoopEnd(const std::vector<Output<Node>>& args, size_t dimension, size_t work_amount, size_t wa_increment,
            std::vector<int64_t> ptr_increments, std::vector<int64_t> finalization_offsets);
    LoopEnd() = delete;
    std::shared_ptr<LoopBegin> get_loop_begin();
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs)  const override;
    const std::vector<int64_t>& get_finalization_offsets() const;
    const std::vector<int64_t>& get_ptr_increments() const;
    void set_finalization_offsets(std::vector<int64_t> offsets);
    void set_ptr_increments(std::vector<int64_t> new_ptr_increments);
    // update_ptr_increments resets non-zero increments to the new_increments. It's used when wa_increment is
    // updated and we need to refresh ptr increments accordingly while respecting the broadcasting pattern
    void update_ptr_increments(int64_t new_increment);
    void set_work_amount(size_t new_work_amount);
    void set_increment(size_t new_increment);
    void set_evaluate_once(bool once);
    // Used to propagate information about Loop structure, needed to simplify some optimizations. For example,
    // to skip pointer increments when outer Loop is empty, and work_amount == vector_size (one inner vector Loop)
    // true by default, the optimizations enabled if it's false;
    bool has_outer_loop;

private:
    std::vector<int64_t> ptr_increments;
    std::vector<int64_t> finalization_offsets;
    size_t loop_io_size;
};

} // namespace op
} // namespace snippets
} // namespace ngraph