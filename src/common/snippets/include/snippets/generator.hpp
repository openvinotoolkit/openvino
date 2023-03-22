// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A file contains public interface for target independent code generator.
 * @file generator.hpp
 */
#pragma once

#include "snippets_isa.hpp"
#include "emitter.hpp"
#include "target_machine.hpp"
#include "lowered_expr.hpp"

namespace ngraph {
namespace snippets {

/**
 * @interface Schedule
 * @brief Return scheduling information and pointer to generated kernel code
 * @ingroup snippets
 */
class Schedule {
public:
    /**
     * @brief Default constructor
     */
    Schedule() : work_size({}), is_flat(false), ptr(nullptr) {}
    /**
     * @brief Default to create schedule out of specific parameters
     * @param ws work size for kernel execution
     * @param f can this kernel be linearided to 1D range
     * @param p pointer to generated code
     */
    Schedule(const ov::PartialShape& ws, bool f, code p) : work_size(ws), is_flat(f), ptr(p) {}
    /**
     * @brief Returns callable instanse of code pointer
     */
    template<typename K> K get_callable() const {
        return reinterpret_cast<K>(const_cast<unsigned char*>(ptr));
    }

    ov::PartialShape work_size {};
    bool is_flat {false};
    code ptr {nullptr};
};
class LoweredExprIR;
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
         LoweringResult(code c, size_t size) : binary_code(c), buffer_scratchpad_size(size) {}
         code binary_code = nullptr;
         size_t buffer_scratchpad_size = 0;
     };
    LoweringResult generate(std::shared_ptr<ov::Model>& m, const LoweringConfig& config, const void* compile_params = nullptr);

    /**
     * @brief gets target machine
     * @return pointer to constant target machine
     */
    std::shared_ptr<const TargetMachine> get_target_machine() const;

protected:
    std::shared_ptr<TargetMachine> target;
    // todo: we need to save lowered code to access compiled brgemm kernels on execution time (normally lowered is destructed by then).
    //  This is temporary solution, remove this when kernel caching is implemented. Don't forget to make generate const method.
    LoweredExprIR lowered_saved;
};

} // namespace snippets
} // namespace ngraph
