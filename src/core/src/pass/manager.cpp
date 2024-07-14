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
}  // namespace

ov::pass::Manager::Manager() : m_pass_config(std::make_shared<PassConfig>()), m_visualize(getenv_visualize_tracing()) {}

ov::pass::Manager::~Manager() = default;

ov::pass::Manager::Manager(std::shared_ptr<ov::pass::PassConfig> pass_config)
    : m_pass_config(std::move(pass_config)),
      m_visualize(getenv_visualize_tracing()) {}

void ov::pass::Manager::set_per_pass_validation(bool new_state) {
    m_per_pass_validation = new_state;
}

namespace {
class stopwatch {
public:
    void start() {
        if (m_active == false) {
            m_active = true;
            m_start_time = m_clock.now();
        }
    }

    void stop() {
        if (m_active == true) {
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
}  // namespace

bool ov::pass::Manager::run_passes(shared_ptr<ov::Model> func) {
    OV_ITT_SCOPED_TASK(ov::itt::domains::core, "pass::Manager::run_passes");

    static bool profile_enabled = ov::util::getenv_bool("OV_PROFILE_PASS_ENABLE");

    size_t index = 0;
    stopwatch pass_timer;
    stopwatch overall_timer;
    overall_timer.start();
    bool pass_applied = false;
    bool function_changed = false;
    bool needs_validate = false;
    for (auto& pass : m_pass_list) {
        if (m_pass_config->is_disabled(pass->get_type_info())) {
            OPENVINO_DEBUG("Pass ", pass->get_name(), " is disabled");
            continue;
        }

        OV_ITT_SCOPE(FIRST_INFERENCE, ov::itt::domains::ov_pass, ov::pass::perf_counters()[pass->get_type_info()]);

        pass_timer.start();

        if (auto matcher_pass = dynamic_pointer_cast<MatcherPass>(pass)) {
            // This checks is to skip the graph transformation when the graph pass relies on
            // static shape but the function state is dynamic.
            if (matcher_pass->get_property(PassProperty::REQUIRE_STATIC_SHAPE) && func->is_dynamic()) {
                OPENVINO_DEBUG("Pass ",
                               pass->get_name(),
                               " requires static shape but the "
                               "model is dynamic. Skipping this transformation");
                continue;
            }
            // GraphRewrite is a temporary container for MatcherPass to make execution
            // on on entire ov::Model
            pass_applied = GraphRewrite(matcher_pass).run_on_model(func);
        } else if (auto function_pass = dynamic_pointer_cast<ModelPass>(pass)) {
            // This checks is to skip the graph transformation when the graph pass relies on
            // static shape but the function state is dynamic.
            if (function_pass->get_property(PassProperty::REQUIRE_STATIC_SHAPE) && func->is_dynamic()) {
                OPENVINO_DEBUG("Pass ",
                               pass->get_name(),
                               " requires static shape but the "
                               "model is dynamic. Skipping this transformation");
                continue;
            }

            if (dynamic_pointer_cast<Validate>(pass)) {
                if (needs_validate) {
                    function_pass->run_on_model(func);
                    needs_validate = false;
                }
            } else {
                pass_applied = function_pass->run_on_model(func);
            }
        }

        if (m_visualize) {
            // visualizations and serializations will be named after the outermost function
            const size_t num_digits_in_pass_index = 3;
            std::string index_str = std::to_string(index);
            index_str = std::string(num_digits_in_pass_index - index_str.length(), '0') + index_str;
            auto base_filename = func->get_name() + std::string("_") + index_str + std::string("_") + pass->get_name();

            if (m_visualize) {
                auto file_ext = "svg";
                pass::VisualizeTree vt(base_filename + std::string(".") + file_ext);
                vt.run_on_model(func);
            }
        }
        index++;
        pass_timer.stop();
        if (profile_enabled) {
            cout << setw(7) << pass_timer.get_milliseconds() << "ms" << (pass_applied ? " + " : "   ")
                 << pass->get_name() << "\n";
        }
        function_changed = function_changed || pass_applied;
        needs_validate = pass_applied;
    }
    if (profile_enabled) {
        cout << "passes done in " << overall_timer.get_milliseconds() << "ms\n";
    }

    return function_changed;
}
