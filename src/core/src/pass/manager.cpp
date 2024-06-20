// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/manager.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "itt.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/visualize_tree.hpp"
#include "openvino/util/env_util.hpp"
#include "openvino/util/log.hpp"
#include "perf_counters.hpp"

using namespace std;

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
bool getenv_visualize_tracing() {
    return ov::util::getenv_bool("OV_ENABLE_VISUALIZE_TRACING");
}

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
    Profiler(bool visualize, bool profile_pass_enable)
        : m_visualize(visualize),
          m_profile_pass_enable(profile_pass_enable) {}

    void start_timer(const std::string& name) {
        if (m_profile_pass_enable) {
            stopwatches[name] = stopwatch();
            stopwatches[name].start();
        }
    }

    void stop_timer(const std::string& name, const std::string& msg) {
        if (m_profile_pass_enable) {
            auto& stopwatch = stopwatches.at(name);
            stopwatch.stop();
            cout << msg << setw(7) << stopwatch.get_milliseconds() << "ms"
                 << "\n";
        }
    }

    void visualize(const shared_ptr<ov::Model>& model, const std::string& name) {
        if (m_visualize) {
            // visualizations and serializations will be named after the outermost function
            const size_t num_digits_in_pass_index = 3;
            std::string index_str = std::to_string(m_index++);
            index_str = std::string(num_digits_in_pass_index - index_str.length(), '0') + index_str;
            auto base_filename = model->get_name() + std::string("_") + index_str + std::string("_") + name;

            auto file_ext = "svg";
            ov::pass::VisualizeTree vt(base_filename + std::string(".") + file_ext);
            vt.run_on_model(model);
        }
    }

private:
    size_t m_index = 0;
    std::unordered_map<std::string, stopwatch> stopwatches;

    bool m_visualize;
    bool m_profile_pass_enable;
};

}  // namespace

ov::pass::Manager::Manager() : m_pass_config(std::make_shared<PassConfig>()), m_visualize(getenv_visualize_tracing()) {}

ov::pass::Manager::~Manager() = default;

ov::pass::Manager::Manager(std::shared_ptr<ov::pass::PassConfig> pass_config)
    : m_pass_config(std::move(pass_config)),
      m_visualize(getenv_visualize_tracing()) {}

void ov::pass::Manager::set_per_pass_validation(bool new_state) {
    m_per_pass_validation = new_state;
}

bool ov::pass::Manager::run_passes(const shared_ptr<ov::Model>& model) {
    OV_ITT_SCOPED_TASK(ov::itt::domains::core, "pass::Manager::run_passes");
    Profiler profiler(m_visualize, ov::util::getenv_bool("OV_PROFILE_PASS_ENABLE"));

    bool pass_applied = false;
    bool model_changed = false;
    bool needs_validate = false;
    const std::string passes_name = "Passes";

    profiler.start_timer(passes_name);
    for (const auto& pass : m_pass_list) {
        const auto& pass_name = pass->get_name();

        if (m_pass_config->is_disabled(pass->get_type_info())) {
            OPENVINO_DEBUG << "Pass " << pass_name << " is disabled";
            continue;
        }

        // This checks if we need to skip the graph transformation when the graph pass relies on
        // static shape but the model state is dynamic.
        if (pass->get_property(PassProperty::REQUIRE_STATIC_SHAPE) && model->is_dynamic()) {
            OPENVINO_DEBUG << "Pass " << pass_name << " requires static shape but the "
                           << "model is dynamic. Skipping this transformation";
            continue;
        }

        OV_ITT_SCOPE(FIRST_INFERENCE, ov::itt::domains::ov_pass, ov::pass::perf_counters()[pass->get_type_info()]);

        profiler.start_timer(pass_name);

        if (auto matcher_pass = dynamic_pointer_cast<MatcherPass>(pass)) {
            // GraphRewrite is a temporary container for MatcherPass to make execution on entire ov::Model
            pass_applied = GraphRewrite(matcher_pass).run_on_model(model);
        } else if (auto model_pass = dynamic_pointer_cast<ModelPass>(pass)) {
            if (dynamic_pointer_cast<Validate>(pass)) {
                if (needs_validate) {
                    needs_validate = false;
                    pass_applied = model_pass->run_on_model(model);
                }
                continue;
            }
            pass_applied = model_pass->run_on_model(model);
        }

        profiler.stop_timer(pass_name, std::string((pass_applied ? " + " : "   ") + pass->get_name()));
        profiler.visualize(model, pass_name);

        model_changed = model_changed || pass_applied;
        needs_validate = needs_validate || pass_applied;
    }

    profiler.stop_timer(passes_name, "All passes done in ");

    return model_changed;
}
