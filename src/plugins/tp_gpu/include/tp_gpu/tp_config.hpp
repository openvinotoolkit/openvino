// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "openvino/runtime/plugin_config.hpp"
#include "tp_gpu/internal_properties.hpp"
#include "tp_gpu/tp_debug.hpp"

// The debug tier keys off ENABLE_TP_GPU_DEBUG_CAPS rather than the stock
// ENABLE_DEBUG_CAPS, so the option table and the call-site macros are driven by
// one switch.  Undefined right after options.inl so the names do not leak.
#ifdef TP_DEBUG_CONFIG
#    define OV_CONFIG_TP_DEBUG_OPTION(...)        OV_CONFIG_DEBUG_OPTION(__VA_ARGS__)
#    define OV_CONFIG_TP_DEBUG_GLOBAL_OPTION(...) OV_CONFIG_DEBUG_GLOBAL_OPTION(__VA_ARGS__)
#else
#    define OV_CONFIG_TP_DEBUG_OPTION(...)
#    define OV_CONFIG_TP_DEBUG_GLOBAL_OPTION(...)
#endif

namespace ov {
namespace tp_gpu {

/// Configuration of the TP_GPU plugin.
///
/// Every knob the plugin has lives in `options.inl` and reaches this class
/// through the same X-macro expansion intel_gpu uses, which is what gives each
/// of them a typed getter, a default, validation, environment and config-file
/// support, and -- for the debug ones -- removal from builds without
/// ENABLE_TP_GPU_DEBUG_CAPS without a single `#ifdef` at the call site.
///
/// Copyable, for the same reason `ov::intel_gpu::ExecutionConfig` is: the base
/// class deletes copying because the options register themselves with the
/// object that owns them, so a copy has to re-register rather than memcpy.
/// Doing that here lets the runtime objects hold a config by value and hand it
/// out by const reference, instead of carrying a hand-written snapshot whose
/// defaults would be a second place to keep in sync.
struct TPConfig : public ov::PluginConfig {
    TPConfig() = default;
    explicit TPConfig(const ov::AnyMap& properties) {
        set_property(properties);
    }

    TPConfig(const TPConfig& other) : TPConfig() {
        assign(other);
    }
    TPConfig& operator=(const TPConfig& other) {
        if (this != &other) {
            assign(other);
        }
        return *this;
    }

    /// The device names of every rank, resolved from TP_SIZE and DEVICE_IDS.
    ///
    /// Empty only when neither was set, which is legal on import: the blob
    /// carries the topology it was produced on and the config only gets a say
    /// when it asks for a different one. `compile_model` requires a topology
    /// and calls `resolve_device_names` instead.
    std::vector<std::string> requested_device_names() const;

    /// Same, but throws when the topology is missing or inconsistent.
    std::vector<std::string> resolve_device_names() const;

    /// Whether `name` is one of this plugin's own options rather than
    /// something meant for the per-rank GPU compilations.
    bool owns(const std::string& name) const;

    /// Drops everything this plugin owns, leaving only what should be
    /// forwarded down to the GPU plugin.
    void erase_own_properties(ov::AnyMap& config) const;

    /// The options a user is allowed to set through the public API.
    std::vector<ov::PropertyName> supported_properties() const;

    /// What was explicitly set, so a per-call config can start from what the
    /// plugin was configured with.
    const ov::AnyMap& user_properties() const {
        return m_user_properties;
    }

    // ---- Debug tier ------------------------------------------------------
    //
    // Named accessors rather than TP_DEBUG_OPT at every call site: the macro
    // and the release-build value then live in exactly one place each, and
    // reading a debug option looks the same as reading any other.

    /// Diagnostics verbosity.  Static because it is a global option: the
    /// collectives inside the intel_gpu plugin reach it without a config.
    static ov::log::Level verbosity() {
        return TP_DEBUG_GLOBAL_OPT(verbose, ov::log::Level::NO);
    }

    ProfilingMode profiling_mode() const {
        return TP_DEBUG_OPT(*this, profiling, ProfilingMode::NONE);
    }

    /// Host-side timings.  Cheap: a few timers around code that runs anyway.
    bool profiling_host() const {
        const auto mode = profiling_mode();
        return mode == ProfilingMode::HOST || mode == ProfilingMode::ALL;
    }

    /// Device-side kernel timestamps.  Forces the host event reset, which puts
    /// the group on the single-threaded rank-0 schedule -- so this measures a
    /// different system than a production run.
    bool profiling_device() const {
        const auto mode = profiling_mode();
        return mode == ProfilingMode::DEVICE || mode == ProfilingMode::ALL;
    }

    /// Rank-0 collectives between two measurement dumps, or 0 when nothing is
    /// being measured.  `per_inference` is what one inference costs in
    /// collective calls and is what an unset period resolves to, so the
    /// default is one report per inference without anyone doing arithmetic.
    std::size_t dump_period(std::size_t per_inference) const {
        if (profiling_mode() == ProfilingMode::NONE) {
            return 0;
        }
        const auto configured = static_cast<std::size_t>(TP_DEBUG_OPT(*this, dump_period, uint64_t{0}));
        if (configured != 0) {
            return configured;
        }
        return per_inference != 0 ? per_inference : 1;
    }

    bool force_sync_collective() const {
        return TP_DEBUG_OPT(*this, force_sync_collective, false);
    }

    bool disable_lm_head_sharding() const {
        return TP_DEBUG_OPT(*this, disable_lm_head_sharding, false);
    }

    bool shard_only() const {
        return TP_DEBUG_OPT(*this, shard_only, false);
    }

    bool skip_collective() const {
        return TP_DEBUG_OPT(*this, skip_collective, false);
    }

private:
    /// Copies values through the option map: the options of `*this` are
    /// already registered with `*this`, so only their values may be taken.
    void assign(const TPConfig& other) {
        m_user_properties = other.m_user_properties;
        m_is_finalized = other.m_is_finalized;
        for (const auto& entry : other.m_options_map) {
            m_options_map.at(entry.first)->set_any(entry.second->get_any());
        }
    }

#include "tp_gpu/options.inl"
};

}  // namespace tp_gpu
}  // namespace ov

#undef OV_CONFIG_TP_DEBUG_OPTION
#undef OV_CONFIG_TP_DEBUG_GLOBAL_OPTION
