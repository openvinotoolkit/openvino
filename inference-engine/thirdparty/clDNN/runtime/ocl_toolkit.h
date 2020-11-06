/*
// Copyright (c) 2016 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "device_info.h"
#include "device_impl.h"
#include "event_impl.h"
#include "ocl_queue_wrapper.h"
#include "ocl_common.h"
#include "configuration.h"
#include "device_cache_reader.h"

#include <memory>
#include <chrono>
#include <string>
#include <vector>
#include <map>
#include <stdexcept>

namespace cldnn {
namespace gpu {

class ocl_error : public std::runtime_error {
public:
    explicit ocl_error(cl::Error const& err);
};

class events_pool;
class gpu_toolkit;

class context_holder {
protected:
    explicit context_holder(std::shared_ptr<gpu_toolkit> context) : _context(context) {}
    virtual ~context_holder() = default;

    const std::shared_ptr<gpu_toolkit>& context() const { return _context; }

    std::shared_ptr<gpu_toolkit> _context;
};


class gpu_toolkit : public std::enable_shared_from_this<gpu_toolkit> {
    friend class context_holder;

protected:
    explicit gpu_toolkit(const device_impl& device_impl,
        const configuration& aconfiguration = configuration());

public:
    static std::shared_ptr<gpu_toolkit> create(const device_impl& device_impl,
        const configuration& cfg = configuration());
    const cl::Context context() const { return _device->get_context(); }
    const cl::Device device() const { return _device->get_device(); }
    const memory_capabilities memory_caps() const { return _device->mem_caps(); }
    const queue_type& queue(uint32_t id) { return get_command_queue(id).queue(); }

    const configuration& get_configuration() const { return _configuration; }
    device_info_internal get_device_info() const { return _device->get_info(); }
    std::shared_ptr<kernel_selector::TuningCache> get_device_cache() const { return _device_cache; }

    inline bool extension_supported(const std::string ext) { return _extensions.find(ext) != std::string::npos; }

    gpu_toolkit(const gpu_toolkit& other) = delete;
    gpu_toolkit(gpu_toolkit&& other) = delete;
    gpu_toolkit& operator=(const gpu_toolkit& other) = delete;
    gpu_toolkit& operator=(gpu_toolkit&& other) = delete;
    std::string single_kernel_name() const { return _configuration.single_kernel_name; }
    bool enabled_single_kernel() const { return single_kernel_name() == "" ? false : true; }

    void set_output_event(uint32_t queue_id, bool out_event);

    event_impl::ptr enqueue_kernel(uint32_t queue_id,
                                   kernel_type const& kern,
                                   cl::NDRange const& global,
                                   cl::NDRange const& local,
                                   std::vector<event_impl::ptr> const& deps);
    event_impl::ptr enqueue_marker(uint32_t queue_id, std::vector<event_impl::ptr> const& deps);
    event_impl::ptr group_events(uint32_t queue_id, std::vector<event_impl::ptr> const& deps);
    void reset_events(uint32_t queue_id);
    event_impl::ptr create_user_event(uint32_t queue_id, bool set);
    void release_events_pool(uint32_t queue_id);
    void release_all_events_pools();

    void flush(uint32_t queue_id);
    void release_pending_memory(uint32_t queue_id);
    void wait_for_events(std::vector<event_impl::ptr> const& events);

    void log(uint64_t id, std::string const& msg);
    bool logging_enabled() const { return !_configuration.log.empty(); }
    bool is_neo_driver() { return _neo_driver; }
    void add_network(uint32_t net_id);
    void remove_network(uint32_t net_id);

private:
    configuration _configuration;
    device_impl::cptr _device;
    bool _neo_driver = false;
    std::map<uint32_t, gpu_queue> _command_queues_w;
    std::shared_ptr<kernel_selector::TuningCache> _device_cache;

    std::string _extensions;

    struct ocl_logger;
    std::unique_ptr<ocl_logger> _logger;

    // returns whether a barrier has been added
    std::ofstream& open_log();

    std::string get_device_version() { return device().getInfo<CL_DEVICE_VERSION>(); }

    gpu_queue& get_command_queue(uint32_t id);

    std::mutex toolkit_mutex;
};

}  // namespace gpu
}  // namespace cldnn
