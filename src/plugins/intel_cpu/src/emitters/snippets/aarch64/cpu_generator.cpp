// Copyright (C) 2020-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpu_generator.hpp"

#include <memory>
#include <openvino/opsets/opset5.hpp>

#include "emitters/plugin/x64/jit_conversion_emitters.hpp"
#include "emitters/plugin/x64/jit_dnnl_ext_emitters.hpp"
#include "emitters/plugin/x64/jit_eltwise_emitters.hpp"
#include "emitters/snippets/cpu_runtime_configurator.hpp"
#include "emitters/snippets/x64/jit_brgemm_copy_b_emitter.hpp"
#include "emitters/snippets/x64/jit_brgemm_emitter.hpp"
#include "emitters/snippets/x64/jit_fill_emitter.hpp"
#include "emitters/snippets/x64/jit_horizon_emitter.hpp"
#include "emitters/snippets/x64/jit_kernel_emitter.hpp"
#include "emitters/snippets/x64/jit_loop_emitters.hpp"
#include "emitters/snippets/x64/jit_memory_emitters.hpp"
#include "emitters/snippets/x64/jit_reg_spill_emitters.hpp"
#include "emitters/snippets/x64/jit_snippets_emitters.hpp"
#include "snippets/snippets_isa.hpp"
#include "transformations/cpu_opset/common/op/swish_cpu.hpp"
#include "transformations/snippets/common/op/fused_mul_add.hpp"
#include "transformations/snippets/x64/op/brgemm_copy_b.hpp"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"
#include "transformations/snippets/x64/op/load_convert.hpp"
#include "transformations/snippets/x64/op/perf_count_rdtsc.hpp"
#include "transformations/snippets/x64/op/store_convert.hpp"
#include "transformations/snippets/x64/pass/lowered/fuse_load_store_and_convert.hpp"

#ifdef SNIPPETS_DEBUG_CAPS
#    include "emitters/snippets/x64/jit_debug_emitter.hpp"
#    include "emitters/snippets/x64/jit_perf_count_chrono_emitters.hpp"
#    include "emitters/snippets/x64/jit_perf_count_rdtsc_emitters.hpp"
#    include "emitters/snippets/x64/jit_segfault_detector_emitter.hpp"
#    include "emitters/snippets/x64/verbose.hpp"
#    include "transformations/snippets/x64/op/perf_count_rdtsc.hpp"
#endif

#ifdef SNIPPETS_LIBXSMM_TPP
#    include "emitters/tpp/x64/jit_brgemm_emitter.hpp"
#    include "emitters/tpp/x64/jit_debug_emitter.hpp"
#    include "emitters/tpp/x64/jit_eltwise_emitters.hpp"
#    include "emitters/tpp/x64/jit_equation_emitter.hpp"
#    include "emitters/tpp/x64/jit_scalar_emitter.hpp"
#    include "transformations/tpp/common/op/brgemm.hpp"
#    include "transformations/tpp/common/op/modifiers.hpp"
#    include "transformations/tpp/x64/op/eltwise.hpp"
#    include "transformations/tpp/x64/op/equation.hpp"
#    include "transformations/tpp/x64/op/reduce.hpp"
#    include "transformations/tpp/x64/op/scalar.hpp"
// Note: for reference implementations
#    include <cmath>
#endif

namespace ov {

#ifdef SNIPPETS_DEBUG_CAPS
static bool is_load_emitter(const intel_cpu::jit_emitter* emitter) {
    bool ret = false;
    if (dynamic_cast<const intel_cpu::jit_load_memory_emitter*>(emitter) ||
        dynamic_cast<const intel_cpu::jit_load_broadcast_emitter*>(emitter)) {
        return true;
    }
    return ret;
}

static bool is_store_emitter(const intel_cpu::jit_emitter* emitter) {
    bool ret = false;
    if (dynamic_cast<const intel_cpu::jit_store_memory_emitter*>(emitter)) {
        return true;
    }
    return ret;
}

static bool is_segfault_detector_emitter(const intel_cpu::jit_emitter* emitter) {
    // default active for typical tensor memory access emitters
    bool ret = false;
    ret = is_load_emitter(emitter) || is_store_emitter(emitter) ||
          dynamic_cast<const intel_cpu::jit_brgemm_emitter*>(emitter) ||
          dynamic_cast<const intel_cpu::jit_brgemm_copy_b_emitter*>(emitter) ||
          dynamic_cast<const intel_cpu::jit_kernel_emitter*>(emitter);
    return ret;
    // use below code to active all emitters for extend usage
    // return !dynamic_cast<const jit_nop_emitter*>(emitter);
}

#    define CREATE_SNIPPETS_EMITTER(e_type, ...)                                                                    \
        {                                                                                                           \
            [this](const snippets::lowered::ExpressionPtr& expr) -> std::shared_ptr<snippets::Emitter> {            \
                auto emitter = std::make_shared<e_type>(h.get(), isa, expr, ##__VA_ARGS__);                         \
                if (debug_config.enable_segfault_detector && is_segfault_detector_emitter(emitter.get())) {         \
                    auto segfault_emitter =                                                                         \
                        std::make_shared<jit_uni_segfault_detector_emitter>(h.get(),                                \
                                                                            isa,                                    \
                                                                            emitter.get(),                          \
                                                                            is_load_emitter(emitter.get()),         \
                                                                            is_store_emitter(emitter.get()),        \
                                                                            expr->get_node()->get_friendly_name()); \
                    return std::make_shared<jit_debug_emitter>(emitter,                                             \
                                                               segfault_emitter,                                    \
                                                               jit_debug_emitter::EmissionLocation::preamble);      \
                }                                                                                                   \
                return emitter;                                                                                     \
            },                                                                                                      \
                [](const std::shared_ptr<ov::Node>& n) -> std::set<std::vector<element::Type>> {                    \
                    return e_type::get_supported_precisions(n);                                                     \
                }                                                                                                   \
        }
#else
#    define CREATE_SNIPPETS_EMITTER(e_type, ...)                                                         \
        {                                                                                                \
            [this](const snippets::lowered::ExpressionPtr& expr) -> std::shared_ptr<snippets::Emitter> { \
                return std::make_shared<e_type>(h.get(), isa, expr, ##__VA_ARGS__);                      \
            },                                                                                           \
                [](const std::shared_ptr<ov::Node>& n) -> std::set<std::vector<element::Type>> {         \
                    return e_type::get_supported_precisions(n);                                          \
                }                                                                                        \
        }
#endif

#define CREATE_DEBUG_TPP_EMITTER(e_type)                                                                  \
    {                                                                                                     \
        [this](const snippets::lowered::ExpressionPtr& expr) -> std::shared_ptr<snippets::Emitter> {      \
            return std::make_shared<DebugTppEmitter>(expr, std::make_shared<e_type>(h.get(), isa, expr)); \
        },                                                                                                \
            [](const std::shared_ptr<ov::Node>& n) -> std::set<std::vector<element::Type>> {              \
                return e_type::get_supported_precisions(n);                                               \
            }                                                                                             \
    }
#define CREATE_CPU_EMITTER(e_type)                                                                                    \
    {                                                                                                                 \
        [this]([[maybe_unused]] const snippets::lowered::ExpressionPtr& expr) -> std::shared_ptr<snippets::Emitter> { \
            return std::make_shared<e_type>(h.get(), isa, expr->get_node());                                          \
        },                                                                                                            \
            []([[maybe_unused]] const std::shared_ptr<ov::Node>& n) -> std::set<std::vector<element::Type>> {         \
                return e_type::get_supported_precisions(n);                                                           \
            }                                                                                                         \
    }

#define CREATE_UNDEFINED_EMITTER(supported_precisions)                                                            \
    {                                                                                                             \
        []([[maybe_unused]] const snippets::lowered::ExpressionPtr& expr) -> std::shared_ptr<snippets::Emitter> { \
            return nullptr;                                                                                       \
        },                                                                                                        \
            []([[maybe_unused]] const std::shared_ptr<ov::Node>& n) -> std::set<std::vector<element::Type>> {     \
                return supported_precisions;                                                                      \
            }                                                                                                     \
    }

class jit_snippet : public dnnl::impl::cpu::x64::jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_snippet)

    ~jit_snippet() override = default;

    jit_snippet() : jit_generator(jit_name()) {}

    void generate() override {}
};

intel_cpu::CPUTargetMachine::CPUTargetMachine(dnnl::impl::cpu::x64::cpu_isa_t host_isa,
                                              ov::intel_cpu::MultiCacheWeakPtr cache)
    : TargetMachine(std::make_shared<CPURuntimeConfigurator>(cache)),
      h(new jit_snippet()),
      isa(host_isa),
      compiled_kernel_cache(std::move(cache)) {
    // data movement
    jitters[op::v0::Parameter::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(intel_cpu::jit_nop_emitter);
    jitters[op::v0::Result::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(intel_cpu::jit_nop_emitter);
    jitters[snippets::op::Buffer::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(intel_cpu::jit_nop_emitter);
    jitters[snippets::op::VectorBuffer::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(intel_cpu::jit_nop_emitter);
    jitters[snippets::op::RankNormalization::get_type_info_static()] =
        CREATE_SNIPPETS_EMITTER(intel_cpu::jit_nop_emitter);
    jitters[snippets::op::Reshape::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(intel_cpu::jit_nop_emitter);
    jitters[snippets::op::Reorder::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(intel_cpu::jit_nop_emitter);

    jitters[snippets::op::Load::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(intel_cpu::jit_load_memory_emitter);
    jitters[snippets::op::LoadReorder::get_type_info_static()] =
        CREATE_SNIPPETS_EMITTER(intel_cpu::jit_load_memory_emitter);
    jitters[snippets::op::BroadcastLoad::get_type_info_static()] =
        CREATE_SNIPPETS_EMITTER(intel_cpu::jit_load_broadcast_emitter);
    jitters[intel_cpu::LoadConvertSaturation::get_type_info_static()] =
        CREATE_SNIPPETS_EMITTER(intel_cpu::jit_load_memory_emitter);
    jitters[intel_cpu::LoadConvertTruncation::get_type_info_static()] =
        CREATE_SNIPPETS_EMITTER(intel_cpu::jit_load_memory_emitter);

    jitters[snippets::op::Store::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(intel_cpu::jit_store_memory_emitter);
    jitters[intel_cpu::StoreConvertSaturation::get_type_info_static()] =
        CREATE_SNIPPETS_EMITTER(intel_cpu::jit_store_memory_emitter);
    jitters[intel_cpu::StoreConvertTruncation::get_type_info_static()] =
        CREATE_SNIPPETS_EMITTER(intel_cpu::jit_store_memory_emitter);

    jitters[snippets::op::Scalar::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(intel_cpu::jit_scalar_emitter);
    jitters[snippets::op::BroadcastMove::get_type_info_static()] =
        CREATE_SNIPPETS_EMITTER(intel_cpu::jit_broadcast_move_emitter);

    jitters[snippets::op::ConvertTruncation::get_type_info_static()] =