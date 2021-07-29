// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cldnn/runtime/engine.hpp"
#include "cldnn/primitives/implementation_desc.hpp"

#include "topology.hpp"

#include <memory>
#include <vector>
#include <string>
#include <map>
#include <utility>

namespace cldnn {

/// @addtogroup cpp_api C++ API
/// @{

/// @defgroup cpp_program Program compilation
/// @{

/// @brief Represents user-provided program build option type.
enum class build_option_type {
    /// @brief Allow primitives fusing during program build (default: false).
    fusing,

    /// @brief Enable implicit reordering for user inputs (default: false).
    optimize_data,

    /// @brief Enable implicit static input reordering for user inputs (default: false).
    allow_static_input_reorder,

    /// @brief Enable debug mode (default: false).
    /// @details This option enforce all program primitives to be accessible as outputs.
    debug,

    /// @brief User selected list of program outputs.
    outputs,

    /// @brief User defined learning parameters.
    learning_config,

    /// @brief Tuning config (default: Tuning is disabled).
    /// @details The tuner will automatically find the optimal kernel/config for each node in the graph,
    /// by running multiple implementations and configurations per node and storing the optimal one in cache.
    /// Expect long execution time in the first run.
    /// After the first run a cache with the tuning results will be created in the path provided.
    /// This cache will be used in the next runs.
    tuning_config,

    /// @brief Specifies a directory to which stages of network compilation should be dumped. (default: empty, i.e. no dumping)
    graph_dumps_dir,
    /// @brief Specifies a directory to which compiled kernels should be cached or can be loaded from. (default: empty, i.e. no caching)
    kernels_cache_dir,
    /// @brief Name for serialization process
    serialize_network,
    load_program,
    force_implementations
};

/// @brief Tuning mode.
enum class tuning_mode {
    /// @brief Tuning is disabled.
    tuning_disabled,

    /// @brief Tuning using the cached data (no on-line tuning for non-existing data).
    tuning_use_cache,

    /// @brief Tuning using the cached data if exist, tune and update cache otherwise.
    tuning_tune_and_cache,

    /// @brief Tuning using the cached data and update tasks.
    /// @details Performs updating tasks like removal of invalid caches, promoting to new format, etc.
    /// No tuning for non-existing data.
    tuning_use_and_update,

    /// @brief Retune the cache data even if it exists.
    tuning_retune_and_cache
};

/// @brief Tuning configuration.
struct tuning_config_options {
    tuning_mode mode;
    std::string cache_file_path;

    tuning_config_options() : mode(tuning_mode::tuning_disabled), cache_file_path("") {}
};

/// @brief Learning parameters.
struct learning_params {
    float momentum = 0.0;
    float weights_decay = 0.0;

    learning_params() : momentum(0.9f), weights_decay(0.0005f) {}
};

/// @brief Represents user-provided program build option.
struct build_option {
    /// @brief Allow primitives fusing during program build (default: false).
    static std::shared_ptr<const build_option> fusing(bool enable = false);

    /// @brief Enable implicit reordering for user inputs (default: false).
    static std::shared_ptr<const build_option> optimize_data(bool enable = false);

    /// @brief Enable implicit reordering for static user inputs (default: false).
    static std::shared_ptr<const build_option> allow_static_input_reorder(bool enable = false);

    /// @brief Enable debug mode (default: false).
    /// @details This option enforce all program primitives to be accessible as outputs.
    static std::shared_ptr<const build_option> debug(bool enable = false);

    /// @brief User selected list of program outputs.
    static std::shared_ptr<const build_option> outputs(const std::vector<primitive_id>& outs);

    /// @brief Tuning configuration (default: false).
    /// @details This option will automatically find the optimal kernel/config for each node in the graph,
    /// by running multiple implementations and configurations per node and storing the optimal one in cache.
    /// Expect long execution time in the first run (unless the cache only mode is enabled).
    /// After the first run a cache with the tuning results will be created in the path provided.
    /// This cache will be used in the next runs.
    static std::shared_ptr<const build_option> tuning_config(
        const tuning_config_options& config = tuning_config_options());

    /// @brief Specifies a directory to which stages of network compilation should be dumped (default: empty, i.e. no dumping)
    static std::shared_ptr<const build_option> graph_dumps_dir(const std::string& dir_path);

    /// @brief Specifies a directory to which compiled kernels should be cached or can be loaded from. (default: empty, i.e. no caching)
    static std::shared_ptr<const build_option> kernels_cache_dir(const std::string& dir_path);

    /// @brief Specifies a name for serialization process.
    static std::shared_ptr<const build_option> serialize_network(const std::string& network_name);
    /// @brief Specifies a name of load_program process.
    static std::shared_ptr<const build_option> load_program(const std::string& network_name);

    /// @brief User defined learning parameters.
    static std::shared_ptr<const build_option> learning_config(const learning_params& params = learning_params());
    /// @brief Specifies user defined implementation details to use.
    static std::shared_ptr<const build_option> force_implementations(implementation_forcing_map forcing);

    virtual ~build_option() = default;

private:
    /// @brief Returns option type represented by this object.
    virtual build_option_type get_type() const = 0;

    friend class build_options;
};

/// @brief @ref build_option specialization for boolean options.
template <build_option_type OptType>
struct build_option_bool : build_option {
    /// @brief Constructs option.
    /// @param value Is option enabled.
    explicit build_option_bool(bool value) : _value(value ? 1 : 0) {}

    /// @brief Is option enabled.
    bool enabled() const { return _value != 0; }

private:
    build_option_type get_type() const override { return OptType; }
    uintptr_t _value;
};

/// @brief @ref build_option specialization for program outputs list.
struct build_option_outputs : build_option {
    /// @brief The list of output ids (names)
    const std::vector<primitive_id> outputs;

    /// @brief Constructs option.
    /// @param outs List of ouput ids (names)
    explicit build_option_outputs(const std::vector<primitive_id>& outs)
        : outputs(outs) {}

private:
    /// @brief Returns build_option_type::outputs.
    build_option_type get_type() const override { return build_option_type::outputs; }

    build_option_outputs(const build_option_outputs& other) = delete;
    build_option_outputs& operator=(const build_option_outputs& other) = delete;
};

/// @brief @ref build_option specialization for learning config.
struct build_option_learning_config : build_option {
    /// @brief Learning parameters.
    const learning_params params;

    /// @brief Constructs learning config build option.
    /// @param learning_params Parameters for learning.
    explicit build_option_learning_config(const learning_params& params)
        : params(params) {}

private:
    /// @brief Returns build_option_type::learning_config.
    build_option_type get_type() const override { return build_option_type::learning_config; }

    build_option_learning_config(const build_option_learning_config& other) = delete;
    build_option_learning_config& operator=(const build_option_learning_config& other) = delete;
};

/// @brief @ref build_option specialization for tuning config.
struct build_option_tuning_config : build_option {
    /// @brief Tuning configuration
    const tuning_config_options config;

    /// @brief Constructs tuning config build option.
    /// @param tuning_config Configuration for the tuning.
    explicit build_option_tuning_config(const tuning_config_options& tuning_config)
        : config(tuning_config) {}

private:
    /// @brief Returns build_option_type::tuning_config.
    build_option_type get_type() const override { return build_option_type::tuning_config; }

    build_option_tuning_config(const build_option_tuning_config& other) = delete;
    build_option_tuning_config& operator=(const build_option_tuning_config& other) = delete;
};

/// @brief @ref build_option specialization for selecting a directory.
template <build_option_type OptType>
struct build_option_directory : build_option {
    const std::string directory_path;

    /// @brief Constructs option.
    /// @param outs List of ouput ids (names)
    explicit build_option_directory(const std::string& dir_path) : directory_path(dir_path) {}

private:
    /// @brief Returns build_option_type::graph_dumps_dir.
    build_option_type get_type() const override { return build_option_type::graph_dumps_dir; }

    build_option_directory(const build_option_directory& other) = delete;
    build_option_directory& operator=(const build_option_directory& other) = delete;
};

/// @brief @ref build_option specialization for selecting a directory.
template <build_option_type OptType>
struct build_option_kernels_cache_dir : build_option {
    const std::string directory_path;

    explicit build_option_kernels_cache_dir(const std::string& dir_path) : directory_path(dir_path) {}

private:
    /// @brief Returns build_option_type::kernels_cache_dir.
    build_option_type get_type() const override { return build_option_type::kernels_cache_dir; }

    build_option_kernels_cache_dir(const build_option_kernels_cache_dir& other) = delete;
    build_option_kernels_cache_dir& operator=(const build_option_kernels_cache_dir& other) = delete;
};

/// @brief @ref build_option specialization for serialization process.
template <build_option_type OptType>
struct build_option_serialization : build_option {
    const std::string serialization_network_name;

    explicit build_option_serialization(const std::string& name) : serialization_network_name(name) {}

private:
    build_option_type get_type() const override { return build_option_type::serialize_network; }

    build_option_serialization(const build_option_serialization& other) = delete;
    build_option_serialization& operator=(const build_option_serialization& other) = delete;
};

/// @brief @ref build_option specialization for load_program process.
template <build_option_type OptType>
struct build_option_load_program : build_option {
    const std::string load_program_name;

    explicit build_option_load_program(const std::string& name) : load_program_name(name) {}

private:
    build_option_type get_type() const override { return build_option_type::load_program; }

    build_option_load_program(const build_option_load_program& other) = delete;
    build_option_load_program& operator=(const build_option_load_program& other) = delete;
};

struct build_option_force_implementations : build_option {
    implementation_forcing_map forcing;

    explicit build_option_force_implementations(implementation_forcing_map _forcing) : forcing(std::move(_forcing)) {}
private:
    build_option_type get_type() const override { return build_option_type::force_implementations; }

    build_option_force_implementations(const build_option_force_implementations& other) = delete;
    build_option_force_implementations& operator=(const build_option_force_implementations& other) = delete;
};

namespace detail {
/// @brief Helper template to convert @ref build_option_type value to particular @ref build_option class.
template <build_option_type OptType>
struct build_option_traits {
    /// @brief @ref build_option object type which represents the particular @p OptType.
    typedef build_option object_type;
    /// @brief Make default @ref build_option corresponding @p OptType
    static std::shared_ptr<const build_option> make_default();
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS
template <>
struct build_option_traits<build_option_type::fusing> {
    typedef build_option_bool<build_option_type::fusing> object_type;
    static std::shared_ptr<const build_option> make_default() { return build_option::fusing(); }
};
template <>
struct build_option_traits<build_option_type::optimize_data> {
    typedef build_option_bool<build_option_type::optimize_data> object_type;
    static std::shared_ptr<const build_option> make_default() { return build_option::optimize_data(); }
};
template <>
struct build_option_traits<build_option_type::allow_static_input_reorder> {
    typedef build_option_bool<build_option_type::allow_static_input_reorder> object_type;
    static std::shared_ptr<const build_option> make_default() { return build_option::allow_static_input_reorder(); }
};
template <>
struct build_option_traits<build_option_type::debug> {
    typedef build_option_bool<build_option_type::debug> object_type;
    static std::shared_ptr<const build_option> make_default() { return build_option::debug(); }
};
template <>
struct build_option_traits<build_option_type::outputs> {
    typedef build_option_outputs object_type;
    static std::shared_ptr<const build_option> make_default() { return build_option::outputs({}); }
};
template <>
struct build_option_traits<build_option_type::learning_config> {
    typedef build_option_learning_config object_type;
    static std::shared_ptr<const build_option> make_default() { return build_option::learning_config(); }
};
template <>
struct build_option_traits<build_option_type::tuning_config> {
    typedef build_option_tuning_config object_type;
    static std::shared_ptr<const build_option> make_default() { return build_option::tuning_config(); }
};
template <>
struct build_option_traits<build_option_type::graph_dumps_dir> {
    typedef build_option_directory<build_option_type::graph_dumps_dir> object_type;
    static std::shared_ptr<const build_option> make_default() { return build_option::graph_dumps_dir({}); }
};
template <>
struct build_option_traits<build_option_type::kernels_cache_dir> {
    typedef build_option_directory<build_option_type::kernels_cache_dir> object_type;
    static std::shared_ptr<const build_option> make_default() { return build_option::kernels_cache_dir({}); }
};
template <>
struct build_option_traits<build_option_type::serialize_network> {
    typedef build_option_serialization<build_option_type::serialize_network> object_type;
    static std::shared_ptr<const build_option> make_default() { return build_option::serialize_network({}); }
};
template <>
struct build_option_traits<build_option_type::load_program> {
    typedef build_option_load_program<build_option_type::load_program> object_type;
    static std::shared_ptr<const build_option> make_default() { return build_option::load_program({}); }
};
template <>
struct build_option_traits<build_option_type::force_implementations> {
    using object_type = build_option_force_implementations;
    static std::shared_ptr<const build_option> make_default() { return build_option::force_implementations({}); }
};

#endif
}  // namespace detail

#ifndef DOXYGEN_SHOULD_SKIP_THIS
inline std::shared_ptr<const build_option> build_option::fusing(bool enable) {
    return std::make_shared<build_option_bool<build_option_type::fusing>>(enable);
}

inline std::shared_ptr<const build_option> build_option::optimize_data(bool enable) {
    return std::make_shared<build_option_bool<build_option_type::optimize_data>>(enable);
}

inline std::shared_ptr<const build_option> build_option::allow_static_input_reorder(bool enable) {
    return std::make_shared<build_option_bool<build_option_type::allow_static_input_reorder>>(enable);
}

inline std::shared_ptr<const build_option> build_option::debug(bool enable) {
    return std::make_shared<build_option_bool<build_option_type::debug>>(enable);
}

inline std::shared_ptr<const build_option> build_option::outputs(const std::vector<primitive_id>& outs) {
    return std::make_shared<build_option_outputs>(outs);
}

inline std::shared_ptr<const build_option> build_option::learning_config(const learning_params& params) {
    return std::make_shared<build_option_learning_config>(params);
}

inline std::shared_ptr<const build_option> build_option::tuning_config(const tuning_config_options& config) {
    return std::make_shared<build_option_tuning_config>(config);
}

inline std::shared_ptr<const build_option> build_option::graph_dumps_dir(const std::string& dir_path) {
    return std::make_shared<build_option_directory<build_option_type::graph_dumps_dir>>(dir_path);
}

inline std::shared_ptr<const build_option> build_option::kernels_cache_dir(const std::string& dir_path) {
    return std::make_shared<build_option_directory<build_option_type::kernels_cache_dir>>(dir_path);
}
inline std::shared_ptr<const build_option> build_option::serialize_network(const std::string& name) {
    return std::make_shared<build_option_serialization<build_option_type::serialize_network>>(name);
}
inline std::shared_ptr<const build_option> build_option::load_program(const std::string& name) {
    return std::make_shared<build_option_load_program<build_option_type::load_program>>(name);
}
inline std::shared_ptr<const build_option> build_option::force_implementations(implementation_forcing_map forcing) {
    return std::make_shared<build_option_force_implementations>(std::move(forcing));
}
#endif

/// @brief Represents program build options list.
class build_options {
public:
    /// @brief Adds or replace option to the options list
    void set_option(std::shared_ptr<const build_option> opt) { add_or_replace_option(opt); }

    /// @brief Adds or replace options to the options list
    template <typename... Args>
    void set_option(std::shared_ptr<const build_option> opt, Args... args) {
        add_or_replace_option(opt);
        set_option(args...);
    }

    /// @brief Constructs build options list from its arguments.
    template <typename... Args>
    explicit build_options(Args... args) {
        set_option(args...);
    }

    /// @brief Returns program build option for @p OptType
    template <build_option_type OptType>
    std::shared_ptr<const typename detail::build_option_traits<OptType>::object_type> get() const {
        using T = typename detail::build_option_traits<OptType>::object_type;
        for (auto& option : _options) {
            if (option->get_type() == OptType)
                return std::static_pointer_cast<const T>(option);
        }
        return std::static_pointer_cast<const T>(detail::build_option_traits<OptType>::make_default());
    }

private:
    friend struct program;
    std::vector<std::shared_ptr<const build_option>> _options;
    void set_option(void) {}

    void add_or_replace_option(std::shared_ptr<const build_option> opt) {
        for (auto& p : _options) {
            if (p->get_type() == opt->get_type()) {
                p = opt;
                return;
            }
        }
        _options.push_back(opt);
    }
};

struct program_impl;

/// @brief Compiled program build from @ref topology by @ref engine
struct program {
    friend struct network;

public:
    /// @brief Builds executable program based on user-defined @p topology by specified @p engine.
    /// @param[in] engine The engine which will be used to build the program.
    /// @param[in] topology The user-defined topology on which the network will be based.
    /// @param[in] options Program build options. See @ref build_option and @ref build_options for details.
    program(engine& engine, const topology& topology, const build_options& options = build_options());

    /// @brief Copy constructor.
    program(const program& other) : _impl(other._impl) { }

    /// @brief Dereferences the counter of the underlying C API @ref cldnn_program handler.
    ~program() { }

    /// @brief Assigns new value by releasing previously referenced C API @ref cldnn_program handler and retaining the one referenced by @p other.
    program& operator=(const program& other) {
        if (_impl == other._impl)
            return *this;
        _impl = other._impl;
        return *this;
    }

    /// @brief Checks whether @p lhs and @p rhs reference the same C API @ref cldnn_program handler
    friend bool operator==(const program& lhs, const program& rhs) { return lhs._impl == rhs._impl; }
    /// @brief Checks whether @p lhs and @p rhs reference different C API @ref cldnn_program handlers
    friend bool operator!=(const program& lhs, const program& rhs) { return !(lhs == rhs); }

    std::shared_ptr<program_impl> get() const { return _impl; }

private:
    std::shared_ptr<program_impl> _impl;

    explicit program(std::shared_ptr<program_impl> impl) : _impl(impl) {
        if (_impl == nullptr)
            throw std::invalid_argument("implementation pointer should not be null");
    }
};
/// @}
/// @}
}  // namespace cldnn
