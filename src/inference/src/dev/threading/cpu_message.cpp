// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/runtime/threading/cpu_message.hpp"

#include <memory>
#include <mutex>
#include <queue>
#include <set>
#include <thread>
#include <vector>

namespace ov {
namespace threading {

MessageManager::MessageManager() {
    _num_sub_streams = 0;
}

MessageManager::~MessageManager() {}

void MessageManager::send_message(const MessageInfo& msg_info) {
    {
        std::lock_guard<std::mutex> lock(_msgMutex);
        _messageQueue.push_back(msg_info);
    }
    _msgCondVar.notify_all();
}

void MessageManager::server_wait() {
    assert(_num_sub_streams);
    MsgType msg_type;
    int count = 0;
    bool isStopped = false;
    while (!isStopped) {
        std::vector<MessageInfo> msgQueue;
        {
            std::unique_lock<std::mutex> lock(_msgMutex);
            _msgCondVar.wait(lock, [&] {
                return !_messageQueue.empty();
            });
            std::swap(_messageQueue, msgQueue);
        }

        for (auto rec_info : msgQueue) {
            msg_type = rec_info.msg_type;
            if (msg_type == CALL_BACK) {  // CALL_BACK
                count++;
                if (count == _num_sub_streams) {
                    count = 0;
                    isStopped = true;
                }
            }
        }
    };
}

void MessageManager::set_num_sub_streams(int num_sub_streams) {
    _num_sub_streams = num_sub_streams;
}

int MessageManager::get_num_sub_streams() {
    return _num_sub_streams;
}

namespace {

class MessageManageHolder {
    std::mutex _mutex;
    std::weak_ptr<MessageManager> _manager;

public:
    MessageManageHolder(const MessageManageHolder&) = delete;
    MessageManageHolder& operator=(const MessageManageHolder&) = delete;

    MessageManageHolder() = default;

    std::shared_ptr<ov::threading::MessageManager> get() {
        std::lock_guard<std::mutex> lock(_mutex);
        auto manager = _manager.lock();
        if (!manager) {
            _manager = manager = std::make_shared<MessageManager>();
        }
        return manager;
    }
};

}  // namespace

std::shared_ptr<MessageManager> message_manager() {
    static MessageManageHolder message_manage;
    return message_manage.get();
}

}  // namespace threading
}  // namespace ov
