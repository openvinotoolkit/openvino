// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/manager.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>

#include "itt.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/visualize_tree.hpp"
#include "openvino/util/common_util.hpp"
#include "openvino/util/env_util.hpp"
#include "openvino/util/log.hpp"
#include "perf_counters.hpp"

#ifdef ENABLE_PROFILING_ITT

namespace ov {
namespace pass {
namespace {
PerfCounters& perf_counters() {
    static PerfCounters counters;
    return counters;
}
}  // namespace
}  // namespace pass
}  // namespace ov

#endif  // ENABLE_PROFILING_ITT

namespace {

/**
 * @brief EnvVar gets the environment variable value by name.
 * It tries to interpret the value as boolean, if it fails then
 * the original string value is stored. This behavior helps us to reduce the number
 * of the additional env variables.
 *
 * Example of usage:
 * if OV_ENABLE_PROFILE_PASS is true, it enables console output.
 * if OV_ENABLE_PROFILE_PASS contains a path to file (string), the out logs
 * will be re-directed to the file.
 */
class EnvVar {
public:
    explicit EnvVar(const std::string& var) {
        const auto& val = ov::util::getenv_string(var.c_str());
        std::set<std::string> off = {"0", "false", "off"};
        std::set<std::string> on = {"1", "true", "on"};

        const auto& val_lower = ov::util::to_lower(val);
        if (off.count(val_lower)) {
            m_is_bool = true;
        } else if (on.count(val_lower)) {
            m_is_bool = true;
            b_value = true;
        } else {
            s_value = val;
        }
    }

    /**
     * @brief This ctor helps to activate/deactivate EnvVar from the code.
     */
    explicit EnvVar(const std::string& var, bool activate) {
        m_is_bool = true;
        b_value = activate;
    }

    bool is_enabled() const {
        return b_value || !s_value.empty();
    }

    bool is_bool() const {
        return m_is_bool;
    }

    const std::string& get_str() const {
        return s_value;
    }

private:
    bool m_is_bool = false;
    bool b_value = false;
    std::string s_value;
};

class stopwatch {
public:
    void start() {
        if (!m_active) {
            m_active = true;
            m_start_time = m_clock.now();
        }
    }

    void stop() {
        if (m_active) {
            auto end_time = m_clock.now();
            m_last_time = end_time - m_start_time;
            m_active = false;
        }
    }

    std::chrono::nanoseconds get_timer_value() const {
        if (m_active) {
            return (m_clock.now() - m_start_time);
        } else {
            return m_last_time;
        }
    }

    size_t get_milliseconds() const {
        return std::chrono::duration_cast<std::chrono::milliseconds>(get_timer_value()).count();
    }

private:
    std::chrono::high_resolution_clock m_clock;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_start_time;
    bool m_active = false;
    std::chrono::nanoseconds m_last_time = std::chrono::high_resolution_clock::duration::zero();
};

class Profiler {
public:
    /**
     * @brief Profiler class helps to analyze Transformations execution times, visualize/serialize ov model after all
     * or for dedicated Transformations.
     *
     *  There are 3 environment variables which can be set for Transformations debugging:
     *
     *  1. OV_ENABLE_PROFILE_PASS - Enables profiling of transformation passes to log their execution times.
     *
     *      Usage: Set this environment variable to "true" to enable visualizations.
     *      Alternatively, specify a file path where the execution times will be saved.
     *
     *      Example:
     *      export OV_ENABLE_PROFILE_PASS=true
     *      export OV_ENABLE_PROFILE_PASS="/path/to/save/profiling/results"
     *
     *  2. OV_ENABLE_VISUALIZE_TRACING - Enables visualization of the model to .svg file after each transformation pass.
     *
     *      Usage: Set this environment variable to "true", "on" or "1" to enable visualization for all Transformations.
     *
     *      Filtering: You can specify filters to control which passes are visualized.
     *      If the variable is set to a specific filter string (e.g., "PassName", "PassName1,PassName2"),
     *      only transformations matching that filter will be visualized. Delimiter is ",".
     *
     *      Example:
     *      export OV_ENABLE_VISUALIZE_TRACING=true
     *      export OV_ENABLE_VISUALIZE_TRACING="Pass1,Pass2,Pass3"
     *
     *  3. OV_ENABLE_SERIALIZE_TRACING - Enables serialization of the model to .xml/.bin after each transformation pass.
     *
     *      Usage: Set this environment variable to "true", "on" or "1" to enable serialization for all Transformations.
     *
     *      Filtering: You can specify filters to control which passes are serialized.
     *      If the variable is set to a specific filter string (e.g., "PassName", "PassName1,PassName2"),
     *      only transformations matching that filter will be serialized. Delimiter is ",".
     *
     *      Example:
     *      export OV_ENABLE_SERIALIZE_TRACING=true
     *      export OV_ENABLE_SERIALIZE_TRACING="Pass1,Pass2,Pass3"
     *
     */
    explicit Profiler(std::string manager_name)
        : m_visualize("OV_ENABLE_VISUALIZE_TRACING"),
          m_serialize("OV_ENABLE_SERIALIZE_TRACING"),
          m_profile_pass("OV_ENABLE_PROFILE_PASS"),
          m_manager_name(std::move(manager_name)) {
        if (m_profile_pass.is_enabled() && !m_profile_pass.is_bool()) {
            m_file.open(m_profile_pass.get_str(), std::ios_base::app);
        }
    }

    ~Profiler() {
        if (m_file.is_open()) {
            m_file.close();
        }
    }

    void start_timer(const std::string& name) {
        if (m_profile_pass.is_enabled()) {
            stopwatches[name] = stopwatch();
            stopwatches[name].start();

            bool is_pass_manager = name == m_manager_name;
            if (is_pass_manager) {
                std::cout << std::setw(25) << std::left;
                std::cout << "PassManager started: " << m_manager_name << std::endl;
                std::cout << std::right;
            }
        }
    }

    void stop_timer(const std::string& name, bool applied) {
        if (m_profile_pass.is_enabled()) {
            auto& stopwatch = stopwatches.at(name);
            stopwatch.stop();

            bool is_pass_manager = name == m_manager_name;
            if (m_profile_pass.is_bool()) {
                std::cout << std::setw(25) << std::left;
                if (is_pass_manager) {
                    std::cout << "PassManager finished: ";
                } else {
                    std::cout << "  ";
                }
                std::cout << std::setw(60) << std::left << name;
                std::cout << std::setw(5) << std::right << stopwatch.get_milliseconds() << "ms "
                          << (applied ? "+" : "-") << std::endl;
            } else if (m_file.is_open()) {
                if (is_pass_manager) {
                    m_file << "m;" << name << ";" << stopwatch.get_timer_value().count() << ";" << (applied ? "1" : "0")
                           << std::endl;
                } else {
                    m_file << "t;" << name << ";" << m_manager_name << ";" << stopwatch.get_timer_value().count() << ";"
                           << (applied ? "1" : "0") << std::endl;
                }
            } else {
                OPENVINO_THROW("The output file for logging transformation statistics is closed. "
                               "Recording of statistics is not possible.");
            }
        }
    }

    void visualize(const std::shared_ptr<ov::Model>& model, const std::string& pass_name) const {
        static size_t viz_index = 0;
        if (m_visualize.is_enabled()) {
            const auto& _visualize = [&]() {
                const auto& file_name = gen_file_name(model->get_name(), pass_name, viz_index++);
                ov::pass::VisualizeTree vt(file_name + ".svg");
                vt.run_on_model(model);
            };

            if (m_visualize.is_bool()) {
                _visualize();
            } else {
                const auto& filter_tokens = split_by_delimiter(m_visualize.get_str(), ',');
                for (const auto& token : filter_tokens) {
                    if (pass_name.find(token) != std::string::npos) {
                        _visualize();
                        return;
                    }
                }
            }
        }
    }

    void serialize(const std::shared_ptr<ov::Model>& model, const std::string& pass_name) const {
        static size_t serialize_index = 0;
        if (m_serialize.is_enabled()) {
            const auto& _serialize = [&]() {
                const auto& file_name = gen_file_name(model->get_name(), pass_name, serialize_index++);
                ov::pass::Serialize serialize(file_name + ".xml", file_name + ".bin");
                serialize.run_on_model(model);
            };

            if (m_serialize.is_bool()) {
                _serialize();
            } else {
                const auto& filter_tokens = split_by_delimiter(m_serialize.get_str(), ',');
                for (const auto& token : filter_tokens) {
                    if (pass_name.find(token) != std::string::npos) {
                        _serialize();
                        return;
                    }
                }
            }
        }
    }

private:
    static std::string gen_file_name(const std::string& model_name, const std::string& pass_name, const size_t idx) {
        std::stringstream name;
        // visualizations and serializations will be named after the outermost function
        const size_t num_digits_in_pass_index = 3;
        std::string index_str = std::to_string(idx);
        index_str = std::string(num_digits_in_pass_index - index_str.length(), '0') + index_str;

        name << model_name << std::string("_") << index_str << std::string("_") << pass_name;
        return name.str();
    }

    static std::vector<std::string> split_by_delimiter(std::string str, char delimiter) {
        std::vector<std::string> res;
        size_t pos = 0;
        while ((pos = str.find(delimiter)) != std::string::npos) {
            res.push_back(str.substr(0, pos));
            str.erase(0, pos + 1);
        }
        if (pos != str.size() - 1) {
            res.push_back(str);
        }
        return res;
    }

    std::unordered_map<std::string, stopwatch> stopwatches;

    EnvVar m_visualize;
    EnvVar m_serialize;
    EnvVar m_profile_pass;

    std::string m_manager_name;
    std::fstream m_file;
};

}  // namespace

ov::pass::Manager::Manager() : m_pass_config(std::make_shared<PassConfig>()) {}

ov::pass::Manager::~Manager() = default;

ov::pass::Manager::Manager(std::string name) : m_pass_config(std::make_shared<PassConfig>()), m_name(std::move(name)) {}

ov::pass::Manager::Manager(std::shared_ptr<ov::pass::PassConfig> pass_config, std::string name)
    : m_pass_config(std::move(pass_config)),
      m_name(std::move(name)) {}

void ov::pass::Manager::set_per_pass_validation(bool new_state) {
    m_per_pass_validation = new_state;
}

bool ov::pass::Manager::run_passes(const std::shared_ptr<ov::Model>& model) {
    OV_ITT_SCOPED_TASK(ov::itt::domains::core, "pass::Manager::run_passes");
    Profiler profiler(m_name);

    bool model_changed = false;
    bool pass_changed_model = false;

    profiler.start_timer(m_name);
    for (const auto& pass : m_pass_list) {
        const auto& pass_name = pass->get_name();

        profiler.start_timer(pass_name);
        pass_changed_model = run_pass(pass, model, pass_changed_model);
        profiler.stop_timer(pass_name, pass_changed_model);

        model_changed = model_changed || pass_changed_model;

        profiler.visualize(model, pass_name);
        profiler.serialize(model, pass_name);
    }
    profiler.stop_timer(m_name, model_changed);

    return model_changed;
}

bool ov::pass::Manager::run_pass(const std::shared_ptr<PassBase>& pass,
                                 const std::shared_ptr<Model>& model,
                                 bool needs_validate) {
    if (m_pass_config->is_disabled(pass->get_type_info())) {
        OPENVINO_DEBUG("Pass ", pass->get_name(), " is disabled.");
        return false;
    }

    // This checks if we need to skip the graph transformation when the graph pass relies on
    // static shape but the model state is dynamic.
    if (pass->get_property(PassProperty::REQUIRE_STATIC_SHAPE) && model->is_dynamic()) {
        OPENVINO_DEBUG("Pass ",
                       pass->get_name(),
                       " requires static shape but the ",
                       "model is dynamic. Skipping this transformation.");
        return false;
    }

    OV_ITT_SCOPE(FIRST_INFERENCE, ov::itt::domains::ov_pass, ov::pass::perf_counters()[pass->get_type_info()]);

    if (auto matcher_pass = std::dynamic_pointer_cast<MatcherPass>(pass)) {
        // GraphRewrite is a temporary container for MatcherPass to make execution on entire ov::Model
        return GraphRewrite(matcher_pass).run_on_model(model);
    } else if (auto model_pass = std::dynamic_pointer_cast<ModelPass>(pass)) {
        if (std::dynamic_pointer_cast<ov::pass::Validate>(model_pass) && !needs_validate) {
            return false;
        }
        return model_pass->run_on_model(model);
    }
    return false;
}
