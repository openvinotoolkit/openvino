// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/command_list.hpp"
#include "ze_common.hpp"
#include "ze_resource.hpp"
#include "ze_stream.hpp"

#ifdef ENABLE_ONEDNN_FOR_GPU
#    include <oneapi/dnnl/dnnl.hpp>
#endif

namespace cldnn::ze {
class ze_command_list : public command_list {
public:
    ze_command_list(const ze_stream& ze_stream, QueueTypes queue_type);
    ~ze_command_list();
#ifdef ENABLE_ONEDNN_FOR_GPU
    dnnl::stream& get_onednn_stream();
#endif

private:
    const ze_stream& _ze_stream;
    QueueTypes _queue_type;
    ze_command_list_resource _cmd_list;
#ifdef ENABLE_ONEDNN_FOR_GPU
    std::shared_ptr<dnnl::stream> _onednn_stream = nullptr;
#endif
};

}  // namespace cldnn::ze
