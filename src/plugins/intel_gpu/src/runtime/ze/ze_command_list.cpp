// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ze_command_list.hpp"
#include "ze_stream.hpp"

#ifdef ENABLE_ONEDNN_FOR_GPU
#    include <oneapi/dnnl/dnnl_ze.hpp>
#endif

namespace cldnn::ze {
ze_command_list::ze_command_list(ze_stream& ze_stream, QueueTypes queue_type)
    : _stream(ze_stream)
    ,_queue_type(queue_type) {
    const auto& engine = ze_stream.get_engine();
    const auto& info = engine.get_device_info();
    auto ctx_handle = engine.get_context().handle();
    auto device_handle = engine.get_device().handle();

    ze_command_list_flags_t flags = queue_type == QueueTypes::out_of_order ? 0 : ZE_COMMAND_LIST_FLAG_IN_ORDER;
    ze_command_list_desc_t command_list_desc = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC,
                                                nullptr,
                                                info.compute_queue_group_ordinal,
                                                flags};
    ze_command_list_handle_t cmd_list_handle = nullptr;
    OV_ZE_EXPECT(ze::zeCommandListCreate(ctx_handle, device_handle, &command_list_desc, &cmd_list_handle));

    _cmd_list = ze_command_list_resource(cmd_list_handle);
}

ze_command_list::~ze_command_list() {
#ifdef ENABLE_ONEDNN_FOR_GPU
    // Destroy OneDNN stream before dropping command list
    _onednn_stream.reset();
#endif
    _cmd_list.drop();
}

#ifdef ENABLE_ONEDNN_FOR_GPU
dnnl::stream& ze_command_list::get_onednn_stream() {
    const auto& engine = _stream.get_engine();
    OPENVINO_ASSERT(_queue_type == QueueTypes::in_order,
                    "[GPU] Can't create onednn stream handle as onednn doesn't support out-of-order queue");
    OPENVINO_ASSERT(engine.get_device_info().vendor_id == INTEL_VENDOR_ID,
                    "[GPU] Can't create onednn stream handle as for non-Intel devices");
    if (!_onednn_stream) {
        _onednn_stream =
            std::make_shared<dnnl::stream>(dnnl::ze_interop::make_stream(engine.get_onednn_engine(),
                                                                         _cmd_list.handle(),
                                                                         _stream.is_profiling_enabled()));
    }

    return *_onednn_stream;
}
#endif

void ze_command_list::reset_impl() {
    OV_ZE_EXPECT(ze::zeCommandListReset(_cmd_list.handle()));
}

void ze_command_list::close_impl() {
    OV_ZE_EXPECT(ze::zeCommandListClose(_cmd_list.handle()));
}

void ze_command_list::enqueue_impl() {
    auto& ze_stream = _stream;
    ze_command_list_handle_t imm_cmd_list = ze_stream.get_immediate_command_list();
    ze_command_list_handle_t enqueued_cmd_list = _cmd_list.handle();
    auto event = std::static_pointer_cast<ze_base_event>(ze_stream.create_base_event());
    auto event_handle = event->get_handle();
    OV_ZE_EXPECT(zeCommandListImmediateAppendCommandListsWithParameters(imm_cmd_list, 1, &enqueued_cmd_list, nullptr, event_handle, 0, nullptr));
    _event = event;
}

void ze_command_list::wait_impl() {
    if (_event) {
        _event->wait();
        _event = nullptr;
    }
}

}  // namespace cldnn::ze
