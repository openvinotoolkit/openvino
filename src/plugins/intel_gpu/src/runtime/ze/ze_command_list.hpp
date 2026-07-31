// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/command_list.hpp"

#include "ze_common.hpp"
#include "ze_resource.hpp"

#ifdef ENABLE_ONEDNN_FOR_GPU
#include <oneapi/dnnl/dnnl.hpp>
#endif

namespace cldnn {
namespace ze {

class ze_command_list : public command_list {
public:
    ze_command_list();
    ~ze_command_list() = default;

    dnnl::stream& get_onednn_stream();

private:
    ze_command_list_resource m_cmd_list;
#ifdef ENABLE_ONEDNN_FOR_GPU
    std::shared_ptr<dnnl::stream> _onednn_stream = nullptr;
#endif
};

}  // namespace ze
}  // namespace cldnn
