// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "openvino/runtime/plugin_config.hpp"
#include "tp/internal_properties.hpp"
#include "tp/tp_debug.hpp"

#ifdef TP_DEBUG_CONFIG
#    define OV_CONFIG_TP_DEBUG_OPTION(...)        OV_CONFIG_DEBUG_OPTION(__VA_ARGS__)
#    define OV_CONFIG_TP_DEBUG_GLOBAL_OPTION(...) OV_CONFIG_DEBUG_GLOBAL_OPTION(__VA_ARGS__)
#else
#    define OV_CONFIG_TP_DEBUG_OPTION(...)
#    define OV_CONFIG_TP_DEBUG_GLOBAL_OPTION(...)
#endif

namespace ov {
namespace tp {

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

    /// The rank devices in rank order, from ov::device::priorities ("TP:GPU.0,GPU.1").
    ///
    /// Empty when no device was listed, which is legal on import: the blob
    /// carries the devices it was produced on and the config only gets a say
    /// when it asks for different ones. `compile_model` requires devices and
    /// calls `resolve_device_names` instead.
    std::vector<std::string> requested_device_names() const;

    /// `requested_device_names()` checked for compilation: at least two
    /// devices, all of one device type.
    std::vector<std::string> resolve_device_names() const;

    /// The device type shared by all `names` ("GPU" for "GPU.0,GPU.1"); throws
    /// when they have different types.
    static std::string device_type(const std::vector<std::string>& names);

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

    bool disable_lm_head_sharding() const {
        return TP_DEBUG_OPT(*this, disable_lm_head_sharding, false);
    }

    bool shard_only() const {
        return TP_DEBUG_OPT(*this, shard_only, false);
    }

    /// The diagnostics the collectives share with this plugin, as the GPU_TP_*
    /// options of a rank context. Only what differs from the default, so a build
    /// whose GPU plugin lacks the options is only refused when one is used.
    ov::AnyMap collective_diagnostics() const {
        ov::AnyMap options;
        if (verbosity() != ov::log::Level::NO) {
            options.emplace("GPU_TP_VERBOSE", verbosity());
        }
        if (profiling_mode() != ProfilingMode::NONE) {
            options.emplace("GPU_TP_PROFILING", ov::Any(profiling_mode()).as<std::string>());
        }
        const auto period = TP_DEBUG_OPT(*this, dump_period, uint64_t{0});
        if (period != 0) {
            options.emplace("GPU_TP_DUMP_PERIOD", period);
        }
        return options;
    }

private:
    void assign(const TPConfig& other) {
        m_user_properties = other.m_user_properties;
        m_is_finalized = other.m_is_finalized;
        for (const auto& entry : other.m_options_map) {
            m_options_map.at(entry.first)->set_any(entry.second->get_any());
        }
    }

#include "tp/options.inl"
};

}  // namespace tp
}  // namespace ov

#undef OV_CONFIG_TP_DEBUG_OPTION
#undef OV_CONFIG_TP_DEBUG_GLOBAL_OPTION
