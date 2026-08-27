// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/command_list.hpp"
#include "ze_common.hpp"
#include "ze_resource.hpp"

#ifdef ENABLE_ONEDNN_FOR_GPU
#    include <oneapi/dnnl/dnnl.hpp>
#endif

namespace cldnn::ze {
struct ze_base_event;
class ze_stream;
class ze_command_list : public command_list {
public:
    using ptr = std::shared_ptr<ze_command_list>;
    ze_command_list(ze_stream& ze_stream, QueueTypes queue_type);
    ~ze_command_list();
    ze_command_list_handle_t handle() const { return _cmd_list.handle(); }
#ifdef ENABLE_ONEDNN_FOR_GPU
    dnnl::stream& get_onednn_stream();
#endif
protected:
    void reset_impl() override;
    void close_impl() override;
    void enqueue_impl() override;
    void wait_impl() override;

private:
    ze_stream& _stream;
    ze_command_list_resource _cmd_list;
    std::shared_ptr<ze_base_event> _event = nullptr;
#ifdef ENABLE_ONEDNN_FOR_GPU
    std::shared_ptr<dnnl::stream> _onednn_stream = nullptr;
#endif
    QueueTypes _queue_type;
};

}  // namespace cldnn::ze
