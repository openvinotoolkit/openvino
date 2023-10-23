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
#include "snippets/lowered/pass/pass.hpp"
#include "snippets/shape_types.hpp"

namespace ov {
namespace snippets {

/**
 * @interface Schedule
 * @brief Return scheduling information and pointer to generated kernel code
 * @ingroup snippets
 */
class Schedule {
public:
    Schedule() = default;
    /**
     * @brief Default to create schedule out of specific parameters
     * @param wd work domain for kernel execution
     * @param p pointer to generated code
     */
    Schedule(const VectorDims& wd, code p) : parallel_exec_domain(wd), ptr(p) {}
    /**
     * @brief Returns callable instanse of code pointer
     */
    template<typename K> K get_callable() const {
        return reinterpret_cast<K>(const_cast<unsigned char*>(ptr));
    }

    VectorDims parallel_exec_domain {};
    code ptr {nullptr};
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
    Generator(const std::shared_ptr<TargetMachine>& t) : target(t), lowered_saved{} {}
    /**
     * @brief Default destructor
     */
    virtual ~Generator() = default;
    /**
    * @interface GeneratorConfig
    * @brief Allows to tweak the lowering process.
    */
    /**
     * @brief virtual method any specific implementation should implement
     * @param m model in canonical for for table-based code generation
     * @param config config with transformation and optimization parameters
     * @param compile_params parameters for generated code
     * @return pointer to generated code
     */
     struct LoweringResult {
         LoweringResult(code c) : binary_code(c) {}
         code binary_code = nullptr;
     };
    LoweringResult generate(lowered::LinearIR& linear_ir, const lowered::Config& config, const void* compile_params = nullptr);

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

protected:
    /**
    * @brief gets register type by specific plugin op type
    * @return register type
    */
    virtual opRegType get_specific_op_reg_type(const std::shared_ptr<ov::Node>& op) const;

    std::shared_ptr<TargetMachine> target;
    // todo: we need to save lowered code to access compiled brgemm kernels on execution time (normally lowered is destructed by then).
    //  This is temporary solution, remove this when kernel caching is implemented. Don't forget to make generate const method.
    lowered::LinearIR lowered_saved;
};

} // namespace snippets
} // namespace ov
