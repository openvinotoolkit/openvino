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
enum MsgType { START_INFER, CALL_BACK, QUIT };

struct MessageInfo {
    MsgType msg_type;
    Task task;
};

class OPENVINO_RUNTIME_API MessageManager {
public:
    MessageManager();

    void send_message(const MessageInfo& msg_info);

    void infer_wait();

    void server_wait();

    void stop_server_thread();

    ~MessageManager();

    void set_num_sub_streams(int num_sub_streams);

private:
    int _num_sub_streams = 0;
    std::thread _serverThread;
    bool _isServerStopped = false;
    std::vector<MessageInfo> _messageQueue;
    std::mutex _msgMutex;
    std::mutex _inferMutex;
    std::condition_variable _msgCondVar;
    std::condition_variable _inferCondVar;
    int call_back_count = 0;
};

OPENVINO_RUNTIME_API std::shared_ptr<MessageManager> message_manager();
}  // namespace threading
}  // namespace ov