// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A file contains public interface for target independent code generator.
 * @file generator.hpp
 */
#pragma once

#include "snippets_isa.hpp"

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/shape_types.hpp"
#include "target_machine.hpp"

namespace ov {
namespace snippets {


class Generator;
/**
 * @interface LoweringResult
 * @brief Holds all relevant information produced during lowering
 * @param compiled_snippet pointer to interface class that encapsulates compiled binary code
 * @param buffer_scratchpad_size the amount of additional memory required by the binary code to execute.
 * Must be allocated and freed by the backend.
 */
class LoweringResult {
    friend class Generator;
    // Some emitters rely on other precompiled kernels.
    // We need to keep the pointers to such emitters alive, so the kernels would still be accessible at runtime.
    std::vector<std::shared_ptr<Emitter>> m_saved_emitters{};

public:
    std::shared_ptr<CompiledSnippet> compiled_snippet = nullptr;
    size_t buffer_scratchpad_size = 0;
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
     * @param domain work domain for kernel execution
     * @param lr lowering result produced during code generation
     */
    Schedule(std::vector<size_t>&& domain, LoweringResult&& lr) : parallel_exec_domain(domain), lowering_result(lr) {}
    Schedule(std::vector<size_t> domain, LoweringResult&& lr) : parallel_exec_domain(std::move(domain)), lowering_result(lr) {}
    /**
     * @brief Returns callable instanse of code pointer
     */
    template<typename K> K get_callable() const {
        return reinterpret_cast<K>(const_cast<unsigned char*>(lowering_result.compiled_snippet->get_code()));
    }

    VectorDims parallel_exec_domain {};
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
    * @interface GeneratorConfig
    * @brief Allows to tweak the lowering process.
    */
    /**
     * @brief generates executable code
     * @param linear_ir lowered IR for code generation
     * @param result variable to hande the result, only compiled_snippet and m_saved_emitters field will be modified
     * @param compile_params compile-time parameters used for code generation
     * @return void
     */
    void generate(lowered::LinearIR& linear_ir, LoweringResult& result, const void* compile_params = nullptr) const;

    /**
     * @brief gets target machine
     * @return pointer to constant target machine
     */
    std::shared_ptr<const TargetMachine> get_target_machine() const;

    /**
    * @interface opRegType
    * @brief Register type of operations
    *        Note that currently there are 4 types of ops:
    *        gpr->gpr: (Parameter, Result, LoopBegin, LoopEnd etc)
    *        gpr->vec: or vec->gpr Load/LoadConvert, Store/StoreConvert, BroadcastLoad etc.
    *        vec->vec: all other "normal" operations that perform calculations on vector registers: Add, BroadcastMove, Power, etc.
    */
    enum opRegType {gpr2gpr, gpr2vec, vec2gpr, vec2vec};
    /**
     * @brief gets register type by op type
     *        TODO: Should be static attribute of emitters
     * @return register type
     */
    opRegType get_op_reg_type(const std::shared_ptr<Node>& op) const;

    virtual std::shared_ptr<Generator> clone() const = 0;

protected:
    /**
    * @brief gets register type by specific plugin op type
    * @return register type
    */
    virtual opRegType get_specific_op_reg_type(const std::shared_ptr<ov::Node>& op) const;
    /**
    * @brief returns true if an emitter can use precompiled kernel.
    * @return bool
    */
    virtual bool uses_precompiled_kernel(const std::shared_ptr<Emitter>& emitter) const { return false; }

    std::shared_ptr<TargetMachine> target;
};

} // namespace snippets
} // namespace ov
