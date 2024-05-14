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

#include "openvino/runtime/common.hpp"
#include "openvino/runtime/threading/istreams_executor.hpp"
#include "openvino/runtime/threading/itask_executor.hpp"

namespace ov {

namespace threading {
enum MsgType { TP, START_INFER, CALL_BACK, REDUCE };

struct MessageInfo {
    MsgType msg_type;
    std::vector<int> rank;
    void* buf;
    Task task;
};
class OPENVINO_RUNTIME_API MessageManage {
public:
    MessageManage();
    void send_message(MessageInfo msg_info);

    std::vector<MessageInfo> wait_message(int cur_rank, int streams_num);

    void infer_wait();

    void reduce_wait(int cur_rank, int streams_num);

    void server_wait(int streams_num);

    ~MessageManage();
private:
    std::thread _serverThread;
    bool _isServerStopped = false;
    std::vector<MessageInfo> _messageQueue;
    std::vector<std::vector<MessageInfo>> _readQueue;
    std::vector<int> _reduceQueue;
    std::mutex _msgMutex;
    std::mutex _readMutex;
    std::mutex _inferMutex;
    std::mutex _reduceMutex;
    std::condition_variable _msgCondVar;
    std::condition_variable _readCondVar;
    std::condition_variable _inferCondVar;
    std::condition_variable _reduceCondVar;
};

OPENVINO_RUNTIME_API std::shared_ptr<MessageManage> message_manager();
}  // namespace threading
}  // namespace ov