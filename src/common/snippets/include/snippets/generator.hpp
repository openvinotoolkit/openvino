// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A file contains public interface for target independent code generator.
 * @file generator.hpp
 */
#pragma once

#include "snippets_isa.hpp"

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/kernel_executor_table.hpp"
#include "snippets/shape_types.hpp"
#include "target_machine.hpp"

namespace ov {
namespace snippets {


class Generator;
/**
 * @interface LoweringResult
 * @brief Holds all relevant information produced during lowering
 * @param compiled_snippet pointer to interface class that encapsulates compiled binary code
 * Must be allocated and freed by the backend.
 */
class LoweringResult {
    friend class Generator;
    // Some emitters rely on other precompiled kernels.
    // We need to keep the pointers to such emitters alive, so the kernels or nodes would still be accessible at runtime.
    std::vector<std::shared_ptr<Emitter>> m_saved_emitters{};

public:
    CompiledSnippetPtr compiled_snippet = nullptr;
    KernelExecutorTablePtr kernel_executor_table = nullptr;
};

/**
 * @interface Schedule
 * @brief Return scheduling information and pointer to generated kernel code
 * @ingroup snippets
 */
class Schedule {
public:
    Schedule() = default;
    /**
     * @brief Create schedule out of specific parameters
     * @param lr lowering result produced during code generation
     */
    Schedule(LoweringResult&& lr) : lowering_result(lr) {}
    /**
     * @brief Returns callable instanse of code pointer
     */
    template<typename K> K get_callable() const {
        return reinterpret_cast<K>(const_cast<unsigned char*>(lowering_result.compiled_snippet->get_code()));
    }

    LoweringResult lowering_result {};
};

/**
 * @interface Generator
 * @brief Target independent code generator interface
 * @ingroup snippets
 */
class Generator {
public:
    /**
     * @brief Default constructor
     */
    Generator(const std::shared_ptr<TargetMachine>& t) : target(t) {}
    /**
     * @brief Default destructor
     */
    virtual ~Generator() = default;
    /**
     * @brief generates executable code
     * @param linear_ir lowered IR for code generation
     * @param compile_params compile-time parameters used for code generation
     * @return variable to handle the result
     */
    LoweringResult generate(const lowered::LinearIRPtr& linear_ir, const void* compile_params = nullptr) const;

    /**
     * @brief gets target machine
     * @return pointer to constant target machine
     */
    std::shared_ptr<const TargetMachine> get_target_machine() const;

    /**
     * @brief gets register type by op type
     *        TODO: Should be static attribute of emitters
     * @return register type
     */
    virtual RegType get_op_out_reg_type(const ov::Output<ov::Node>& out) const;

    virtual std::shared_ptr<Generator> clone() const = 0;

protected:
    /**
    * @brief gets register type by specific plugin op type
    * @return register type
    */
    virtual RegType get_specific_op_out_reg_type(const ov::Output<Node>& out) const;
    /**
    * @brief returns true if an emitter can use precompiled kernel.
    * @return bool
    */
    virtual bool uses_precompiled_kernel(const std::shared_ptr<Emitter>& emitter) const { return false; }

    std::shared_ptr<TargetMachine> target;
};

} // namespace snippets
} // namespace ov
