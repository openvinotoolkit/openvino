#pragma once

#include <map>
#include <memory>
#include <vector>

#include "openvino/core/core_visibility.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/node.hpp"

namespace ov {
namespace pass {

using NodeMapping = std::map<std::shared_ptr<ov::Node>, std::weak_ptr<ov::Node>>;

/*
 * PassTrackerState is a shared object between PassTracker instances which
 * is needed to control execution process.
 */
class PassTrackerState {
public:
    PassTrackerState() = default;

    void enable_tracker() {
        reset_model_state();
        m_is_enabled = true;
    }

    void disable_tracker() {
        m_is_enabled = false;
    }

    size_t get_pass_tracker_id() {
        // Returns PassTracker id. Initially it returned unique ids but
        // it is no longer required as we don't need nested PassTrackers.
        return m_global_pass_tracker_id;
    }

    void set_is_active(bool flag) {
        // Acquires pass tracker state, so all other PassTracker instances
        // that were created during this period won't be active. This is needed
        // to avoid nested PassTrackers.
        m_is_active = flag;
    }

    bool is_active() const { return m_is_active; }

    void set_is_enabled(bool flag) {
        // Generic flag to enable/disable pass tracking.
        m_is_enabled = flag;
    }

    bool is_enabled() const { return m_is_enabled; }

    void set_pass_name(std::string pass_name) {
        // Pass name which has acquired pass tracker state.
        m_pass_name = std::move(pass_name);
    }

    const std::string & pass_name() const { return m_pass_name; }

    NodeMapping & model_to_orig() {
        // Returns a node mapping between cloned model and original model, so
        // we can easily detect which nodes were eliminated.
        return m_model_to_orig;
    }

    void set_model(std::shared_ptr<ov::Model> model) {
        m_model = std::move(model);
    }

    std::shared_ptr<ov::Model> model() const {
        // Returns a model which is a clone* of original model.
        return m_model;
    }

private:
    void reset_model_state()  {
        // Resets pass tracker state to its default values. Usually required when
        // PassTracker was used more than once and there is a between when graph
        // could possibly change without PassTracker enabled, so it is better to reset state.
        m_model = nullptr;
        m_model_to_orig.clear();

        m_pass_name.clear();

        m_is_enabled = false;
        m_is_active = false;
    }

    // General flag to enable/disable pass tracking.
    bool m_is_enabled = false;

    // Flag that indicates that tracker state was acquired.
    bool m_is_active = false;

    // Global pass tracker id, probably can be removed as we
    // don't want to support nested PassTrackers which is compute expensive
    // and useless for human analysis.
    size_t m_global_pass_tracker_id = 0;

    // Names of the pass which is being tracked.
    std::string m_pass_name;

    // Mapping between nodes in cloned model and original model.
    NodeMapping m_model_to_orig;

    // Cloned model which represents the previous state of original model.
    std::shared_ptr<ov::Model> m_model;
};

/*
 * PassTracker takes original model as an input and tracks changes inside the model (new nodes, eliminated nodes) and
 * it keeps track of two model states: before transformation and after in efficient way.
 */
class PassTracker {
public:
    PassTracker(std::shared_ptr<ov::Model> orig_model, std::string pass_name);

    ~PassTracker();

    static PassTrackerState & get_state() {
        return m_state;
    }

private:
    bool m_skip_tracker = false;

    size_t m_local_pass_tracker_id;

    std::string m_pass_name;
    std::shared_ptr<ov::Model> m_orig_model;

    static PassTrackerState m_state;
};
} // namespace pass
} // namespace ov