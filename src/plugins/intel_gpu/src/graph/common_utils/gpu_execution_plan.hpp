// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

#include "common_utils/gpu_kernel_lifecycle.hpp"
#include "intel_gpu/runtime/stream.hpp"
#include "openvino/core/except.hpp"

namespace cldnn {

enum class gpu_dispatch_dependency_policy {
    external,
    previous,
};

struct gpu_execution_completion_policy {
    bool request_each_dispatch = false;
    bool aggregate_dispatch_events = false;
};

struct gpu_dispatch_plan {
    size_t kernel_index = 0;
    gpu_dispatch_dependency_policy dependency = gpu_dispatch_dependency_policy::previous;
    bool skip_execution = false;
};

/// A per-dispatch, non-owning descriptor paired with invocation-specific resources.
///
/// Providers construct this value directly from a primitive instance. The plan stays
/// independent of primitive and backend types while retaining the standard GPU kernel
/// argument contract.
struct gpu_dispatch_binding {
    const kernel_arguments_desc* descriptor = nullptr;
    kernel_arguments_data arguments;
};

/// Precomputed backend-neutral execution metadata for a logical 1..N GPU kernel sequence.
///
/// Dispatch topology and scratch capacity are prepared at compile/update time. execute()
/// performs direct indexed access and does not use strings, maps, virtual argument
/// resolvers, or grow its own vectors in the inference path.
class gpu_execution_plan final {
public:
    gpu_execution_plan() = default;

    explicit gpu_execution_plan(size_t dispatch_count, gpu_execution_completion_policy completion = {}) : _completion(completion) {
        resize(dispatch_count);
    }

    void resize(size_t dispatch_count) {
        _dispatches.resize(dispatch_count);
        for (size_t index = 0; index < dispatch_count; ++index) {
            _dispatches[index].kernel_index = index;
        }
        _dependency_scratch.reserve(1);
        _event_scratch.reserve(dispatch_count);
    }

    size_t size() const noexcept {
        return _dispatches.size();
    }

    bool empty() const noexcept {
        return _dispatches.empty();
    }

    gpu_dispatch_plan& operator[](size_t index) {
        return _dispatches.at(index);
    }

    const gpu_dispatch_plan& operator[](size_t index) const {
        return _dispatches.at(index);
    }

    void set_completion_policy(gpu_execution_completion_policy completion) noexcept {
        _completion = completion;
    }

    void suppress_zero_size(bool suppress) noexcept {
        _zero_size = suppress;
    }

    bool zero_size_suppressed() const noexcept {
        return _zero_size;
    }

    template <typename BindingProvider>
    event::ptr execute(stream& command_stream,
                       const gpu_kernel_lifecycle& lifecycle,
                       const std::vector<event::ptr>& external_dependencies,
                       bool request_completion,
                       BindingProvider&& provide_binding) {
        return execute_with(command_stream,
                            lifecycle,
                            external_dependencies,
                            request_completion,
                            std::forward<BindingProvider>(provide_binding),
                            [&command_stream](size_t,
                                              kernel& selected_kernel,
                                              const kernel_arguments_desc& descriptor,
                                              const kernel_arguments_data& arguments,
                                              const std::vector<event::ptr>& dependencies,
                                              bool dispatch_completion) {
                                return command_stream.enqueue_kernel(selected_kernel, descriptor, arguments, dependencies, dispatch_completion);
                            });
    }

    template <typename BindingProvider, typename DispatchExecutor>
    event::ptr execute_with(stream& command_stream,
                            const gpu_kernel_lifecycle& lifecycle,
                            const std::vector<event::ptr>& external_dependencies,
                            bool request_completion,
                            BindingProvider&& provide_binding,
                            DispatchExecutor&& execute_dispatch) {
        OPENVINO_ASSERT(lifecycle.size() >= required_kernel_count(), "[GPU] Execution plan references a kernel that was not initialized");

        const auto last_dispatch = find_last_enabled_dispatch();
        if (_zero_size || last_dispatch == _dispatches.size()) {
            return command_stream.aggregate_events(external_dependencies, external_dependencies.size() > 1);
        }

        _event_scratch.clear();
        event::ptr previous_event;
        for (size_t dispatch_index = 0; dispatch_index <= last_dispatch; ++dispatch_index) {
            const auto& dispatch = _dispatches[dispatch_index];
            if (dispatch.skip_execution) {
                continue;
            }

            auto binding = provide_binding(dispatch_index);
            OPENVINO_ASSERT(binding.descriptor != nullptr, "[GPU] Execution plan received an empty kernel argument descriptor");

            const std::vector<event::ptr>* dependencies = &external_dependencies;
            if (dispatch.dependency == gpu_dispatch_dependency_policy::previous && previous_event != nullptr) {
                _dependency_scratch.clear();
                _dependency_scratch.push_back(previous_event);
                dependencies = &_dependency_scratch;
            }

            const bool is_final_dispatch = dispatch_index == last_dispatch;
            const bool dispatch_completion = _completion.request_each_dispatch ? request_completion : (request_completion && is_final_dispatch);
            auto completion = execute_dispatch(dispatch_index,
                                               *lifecycle.at(dispatch.kernel_index),
                                               *binding.descriptor,
                                               binding.arguments,
                                               *dependencies,
                                               dispatch_completion);
            previous_event = completion;
            if (_completion.aggregate_dispatch_events && completion != nullptr) {
                _event_scratch.push_back(completion);
            }
        }

        if (_completion.aggregate_dispatch_events) {
            if (_event_scratch.empty()) {
                return command_stream.aggregate_events(external_dependencies, external_dependencies.size() > 1);
            }
            // Completion was already requested from each selected dispatch. Do not replace those
            // backend events with an output marker: an in-order stream may represent such a
            // marker as an already-completed user event.
            return command_stream.aggregate_events(_event_scratch, _event_scratch.size() > 1);
        }
        return previous_event;
    }

private:
    size_t required_kernel_count() const noexcept {
        size_t count = 0;
        for (const auto& dispatch : _dispatches) {
            count = std::max(count, dispatch.kernel_index + 1);
        }
        return count;
    }

    size_t find_last_enabled_dispatch() const noexcept {
        if (_zero_size) {
            return _dispatches.size();
        }
        for (size_t index = _dispatches.size(); index > 0; --index) {
            if (!_dispatches[index - 1].skip_execution) {
                return index - 1;
            }
        }
        return _dispatches.size();
    }

    std::vector<gpu_dispatch_plan> _dispatches;
    gpu_execution_completion_policy _completion;
    bool _zero_size = false;
    std::vector<event::ptr> _dependency_scratch;
    std::vector<event::ptr> _event_scratch;
};

}  // namespace cldnn
