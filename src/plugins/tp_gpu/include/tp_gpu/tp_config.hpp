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

#ifdef TP_DEBUG_CONFIG
#    define OV_CONFIG_TP_DEBUG_OPTION(...)        OV_CONFIG_DEBUG_OPTION(__VA_ARGS__)
#    define OV_CONFIG_TP_DEBUG_GLOBAL_OPTION(...) OV_CONFIG_DEBUG_GLOBAL_OPTION(__VA_ARGS__)
#else
#    define OV_CONFIG_TP_DEBUG_OPTION(...)
#    define OV_CONFIG_TP_DEBUG_GLOBAL_OPTION(...)
#endif

namespace ov {
namespace tp_gpu {

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

    /// A version of `requested_device_names()` with stricter validation
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

    static ov::log::Level verbosity() {
        return TP_DEBUG_GLOBAL_OPT(verbose, ov::log::Level::NO);
    }

    ProfilingMode profiling_mode() const {
        return TP_DEBUG_OPT(*this, profiling, ProfilingMode::NONE);
    }

    bool profiling_host() const {
        const auto mode = profiling_mode();
        return mode == ProfilingMode::HOST || mode == ProfilingMode::ALL;
    }

    bool profiling_device() const {
        const auto mode = profiling_mode();
        return mode == ProfilingMode::DEVICE || mode == ProfilingMode::ALL;
    }

    std::size_t dump_period() const {
        if (profiling_mode() == ProfilingMode::NONE) {
            return 0;
        }
        const auto configured = static_cast<std::size_t>(TP_DEBUG_OPT(*this, dump_period, uint64_t{0}));
        return configured != 0 ? configured : 1;
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
