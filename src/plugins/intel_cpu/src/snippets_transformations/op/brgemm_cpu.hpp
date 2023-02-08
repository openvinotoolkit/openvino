// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/op/brgemm.hpp"
#include "brgemm_copy_b.hpp"

namespace ov {
namespace intel_cpu {

/* Class diagram:
 *  Brgemm <----- BrgemmCPU
 *            \
 *             -- BrgemmWithRepackingCPU <----- BrgemmIndependentCPU
 *                                          \
 *                                           -- BrgemmWithScratchCPU <----- BrgemmWithCompensationsCPU
 *                                                                      \
 *                                                                       -- BrgemmAMXCPU
 * 
 * Notes:
 *      - BrgemmCPU - class for FP32 calculations
 *      - BrgemmWithRepackingCPU - `base` class for Brgemm that requires data repacking (BrgemmCopyB on 2nd input)
 *      - BrgemmIndependentCPU - class for U8|I8 and BF16|BF16 calculations. Has 2 inputs and requires BrgemmCopyB on 2nd input
 *      - BrgemmWithScratchCPU - `base` class for Brgemm that requires scratchpad on new 3rd input
 *      - BrgemmWithCompensationsCPU - class for Brgemm that requires comensations on 3rd input
 *      - BrgemmAMXCPU - class for Brgemm for target machine with AMX support
 *
 * The real examples are in plugin-specific pass `BrgemmToBrgemmCPU`
 */


/**
 * @interface BrgemmCPU
 * @brief BrgemmCPU is a batch-reduced matrix multiplication with the support of arbitrary strides between matrices rows
 *        with support only of FP32 precision on inputs
 * @ingroup snippets
 */
class BrgemmCPU : public ngraph::snippets::op::Brgemm {
public:
    OPENVINO_OP("BrgemmCPU", "SnippetsOpset", ngraph::snippets::op::Brgemm);
    BrgemmCPU(const Output<Node>& A, const Output<Node>& B,
              const bool transposed_a = false, const bool transposed_b = false,
              const size_t offset_a = 0, const size_t offset_b = 0, const size_t offset_c = 0);
    BrgemmCPU() = default;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
};

/**
 * @interface BrgemmWithRepackingCPU
 * @brief BrgemmWithRepackingCPU is base class with common interface for Brgemm that requires BrgemmCopyB on second input.
 * @ingroup snippets
 */
class BrgemmWithRepackingCPU : public ngraph::snippets::op::Brgemm {
public:
    OPENVINO_OP("BrgemmWithRepackingCPU", "SnippetsOpset", ngraph::snippets::op::Brgemm);
    BrgemmWithRepackingCPU() = default;

    std::shared_ptr<BrgemmCopyBBase> get_brgemm_copy() const;

protected:
    void validate_output();
    void validate_and_infer_types() override {}
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override { return nullptr; }
};

/**
 * @interface BrgemmIndependentCPU
 * @brief BrgemmIndependentCPU is a batch-reduced matrix multiplication with the support of arbitrary strides between matrices rows
 *        with support of U8|I8 and BF16|BF16 on inputs. Requires BrgemmCopyB but doesn't require compensations or scratchpad.
 * @ingroup snippets
 */
class BrgemmIndependentCPU : public BrgemmWithRepackingCPU {
public:
    OPENVINO_OP("BrgemmIndependentCPU", "SnippetsOpset", BrgemmWithRepackingCPU);
    BrgemmIndependentCPU(const Output<Node>& A, const Output<Node>& B,
                         const bool transposed_a = false, const bool transposed_b = false,
                         const size_t offset_a = 0, const size_t offset_b = 0, const size_t offset_c = 0);
    BrgemmIndependentCPU() = default;

    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
};

/**
 * @interface BrgemmWithScratchCPU
 * @brief BrgemmWithScratchCPU is base class with common interface for Brgemm that requires third input for scratchpad.
 * @ingroup snippets
 */
class BrgemmWithScratchCPU : public BrgemmWithRepackingCPU {
public:
    OPENVINO_OP("BrgemmWithScratchCPU ", "SnippetsOpset", BrgemmWithRepackingCPU);
    BrgemmWithScratchCPU() = default;

    size_t get_offset_scratch() const { return get_input_port_descriptor(2).m_offset; }

protected:
    BrgemmWithScratchCPU(const Output<Node>& A, const Output<Node>& B, const Output<Node>& scratch,
                         const bool transposed_a = false, const bool transposed_b = false,
                         const size_t offset_a = 0, const size_t offset_b = 0, const size_t offset_scratch = 0, const size_t offset_c = 0);

    void validate_and_infer_types() override {};
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override { return nullptr; }
};

/**
 * @interface BrgemmWithCompensationsCPU
 * @brief BrgemmWithCompensationCPU is a batch-reduced matrix multiplication with the support of arbitrary strides between matrices rows
 *        with support compensations
 *        Note: Brgemm needs 3rd input for compensations in cases when 1st input is I8 and target machine doesn't support AMX
 * @ingroup snippets
 */
class BrgemmWithCompensationsCPU : public BrgemmWithScratchCPU {
public:
    OPENVINO_OP("BrgemmWithCompensationsCPU", "SnippetsOpset", BrgemmWithScratchCPU);
    BrgemmWithCompensationsCPU(const Output<Node>& A, const Output<Node>& B, const Output<Node>& scratch,
                               const bool transposed_a = false, const bool transposed_b = false,
                               const size_t offset_a = 0, const size_t offset_b = 0, const size_t offset_scratch = 0, const size_t offset_c = 0);
    BrgemmWithCompensationsCPU() = default;

    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
};

/**
 * @interface BrgemmAMXCPU
 * @brief BrgemmAMXCPU is a batch-reduced matrix multiplication with the support of arbitrary strides between matrices rows
 *        with support AMX instructions
 * @ingroup snippets
 */
class BrgemmAMXCPU : public BrgemmWithScratchCPU {
public:
    OPENVINO_OP("BrgemmAMXCPU", "SnippetsOpset", BrgemmWithScratchCPU);
    BrgemmAMXCPU(const Output<Node>& A, const Output<Node>& B, const Output<Node>& scratch,
                 const bool transposed_a = false, const bool transposed_b = false,
                 const size_t offset_a = 0, const size_t offset_b = 0, const size_t offset_scratch = 0, const size_t offset_c = 0);
    BrgemmAMXCPU() = default;

    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
};


} // namespace intel_cpu
} // namespace ov
