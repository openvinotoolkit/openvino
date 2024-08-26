// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief OpenVINO Runtime Executor Manager
 * @file openvino/runtime/threading/executor_manager.hpp
 */

#pragma once

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <set>
#include <thread>
#include <vector>
#include <assert.h>

#include "openvino/runtime/common.hpp"
#include "openvino/runtime/threading/istreams_executor.hpp"
#include "openvino/runtime/threading/itask_executor.hpp"


namespace ov {

namespace threading {
enum MsgType {CALL_BACK};

struct MessageInfo {
    MsgType msg_type;
};

class OPENVINO_RUNTIME_API MessageManager {
public:
    MessageManager();

    void send_message(const MessageInfo& msg_info);

    void server_wait();

    ~MessageManager();

    void set_num_sub_streams(int num_sub_streams);

    int get_num_sub_streams();
private:
    int _num_sub_streams = 0;
    std::vector<MessageInfo> _messageQueue;
    std::mutex _msgMutex;
    std::condition_variable _msgCondVar;
};

OPENVINO_RUNTIME_API std::shared_ptr<MessageManager> message_manager();
}  // namespace threading
}  // namespace ov