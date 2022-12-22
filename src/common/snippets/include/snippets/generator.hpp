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
#include "ie_precision.hpp"

namespace ngraph {
namespace snippets {

auto getRegisters(std::shared_ptr<ngraph::Node>& n) -> ngraph::snippets::RegInfo;

typedef std::pair<std::function<std::shared_ptr<Emitter>(std::shared_ptr<ngraph::Node>)>,
                  std::set<std::vector<InferenceEngine::Precision>>> jitters_value;
/**
 * @interface TargetMachine
 * @brief Base class Target machine representation. Target derives from this class to provide generator information about supported emitters
 * @ingroup snippets
 */
class TargetMachine {
public:
    /**
     * @brief checks if target is natively supported
     * @return true, if supported
     */
    virtual bool is_supported() const = 0;

    /**
     * @brief finalizes code generation
     * @return generated kernel binary
     */
    virtual code get_snippet() const = 0;

    /**
     * @brief gets number of lanes supported by target's vector ISA
     * @return number of lanes
     */
    virtual size_t get_lanes() const = 0;


    /**
     * @brief called by generator to all the emitter for a target machine
     * @return a map by node's type info with callbacks to create an instance of emitter for corresponding operation type
     */
    std::function<std::shared_ptr<Emitter>(std::shared_ptr<ngraph::Node>)> get(const ngraph::DiscreteTypeInfo type) const {
        auto jitter = jitters.find(type);
        if (jitter == jitters.end()) {
            throw ngraph_error(std::string("Target code emitter is not available for ") + type.name + " operation.");
        }
        return jitter->second.first;
    }

     std::set<std::vector<InferenceEngine::Precision>> get_supported_precisions(const ngraph::DiscreteTypeInfo type) const {
        auto jitter = jitters.find(type);
        if (jitter == jitters.end()) {
            throw ngraph_error(std::string("Target code emitter is not available for ") + type.name + " operation.");
        }
        return jitter->second.second;
    }

    /**
     * @brief checks if emitter for a specific operation is supported
     * @return true, if supported
     */
    bool has(const ngraph::DiscreteTypeInfo type) const {
        return jitters.find(type) != jitters.end();
    }
    virtual ~TargetMachine() = default;

protected:
    std::map<
        const ngraph::DiscreteTypeInfo, 
        jitters_value> jitters;
};

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
    class GeneratorConfig {
    public:
        // True if the lowered Emitters need to be accessed during runtime. Normally they're destroyed after code emission.
        bool m_save_lowered_code = false;
        // True if we can optimize tails for single evaluation during code generation
        // More details with optimization examples you can see in generate() method
        // For example, tails with Buffer ops doesn't support single evaluation optimizations
        //              because of that we should always reset memory pointer using finalization offsets
        //              after data storing to Buffer
        bool m_optimize_single_evaluation = true;
        // True if we should check runtime info for nodes to call specific needed transformations
        bool m_need_fill_tail_register = false;
    };
    /**
     * @brief virtual method any specific implementation should implement
     * @param m model in canonical for for table-based code generation
     * @param config config with transformation and optimization parameters
     * @param compile_params parameters for generated code
     * @return pointer to generated code
     */
    code generate(std::shared_ptr<ov::Model>& m, const GeneratorConfig& config, const void* compile_params = nullptr);

    /**
     * @brief gets target machine
     * @return pointer to constant target machine
     */
    std::shared_ptr<const TargetMachine> get_target_machine() const;

protected:
    std::shared_ptr<TargetMachine> target;
    // todo: we need to save lowered code to access compiled brgemm kernels on execution time (normally lowered is destructed by then).
    //  This is temporary solution, remove this when kernel caching is implemented. Don't forget to make generate const method.
    std::vector<AllocatedEmitter> lowered_saved;
};

} // namespace snippets
} // namespace ngraph
