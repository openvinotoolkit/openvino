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
#include "openvino/runtime/compiled_model.hpp"
#include "openvino/runtime/iasync_infer_request.hpp"

namespace ov {

namespace threading {
enum MsgType { START_INFER, TENSOR_PARALLEL, CALL_BACK, REDUCE, QUIT };

struct MessageInfo {
    MsgType msg_type;
    std::vector<int> rank;
    void* buf;
    Task task;
};

struct MemoryInfo {
    void* send_buf;
    std::shared_ptr<void> buf;
    bool flag;
    bool last_used;
};

class OPENVINO_RUNTIME_API MessageManager {
public:
    MessageManager();

    void send_message(const MessageInfo& msg_info);

    std::vector<MessageInfo> wait_message(int stream_id);

    void infer_wait();

    void reduce_wait(int stream_id);

    void server_wait();

    void stop_server_thread();

    void clear();

    ~MessageManager();

    void set_sub_compiled_models(std::vector<std::shared_ptr<ov::ICompiledModel>> models);
    std::vector<std::shared_ptr<ov::ICompiledModel>> get_sub_compiled_models();

    void set_sub_infer_requests(std::vector<std::shared_ptr<ov::IAsyncInferRequest>> requests);
    std::vector<std::shared_ptr<ov::IAsyncInferRequest>> get_sub_infer_requests();

    int get_num_sub_streams();

    int get_memory_id(int sub_stream_id);

    void set_memory_used(int memory_id, int sub_stream_id);

    std::vector<std::vector<MemoryInfo>> _memorys_table;
    std::vector<int> _use_count;
    std::mutex _flagMutex;

private:
    int _num_sub_streams;
    std::vector<std::shared_ptr<ov::ICompiledModel>> _sub_compiled_models;
    std::vector<std::shared_ptr<ov::IAsyncInferRequest>> _sub_infer_requests;
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

OPENVINO_RUNTIME_API std::shared_ptr<MessageManager> message_manager();
}  // namespace threading
}  // namespace ov