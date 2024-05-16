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

MessageManager::MessageManager() {}

MessageManager::~MessageManager() {}

void MessageManager::send_message(const MessageInfo& msg_info) {
    {
        std::lock_guard<std::mutex> lock(_msgMutex);
        _messageQueue.push_back(msg_info);
        // std::cout << "send : " << msg_info.msg_type << ", " << (msg_info.rank.size() > 0 ? msg_info.rank[0] : -1)
        //           << "\n";
    }
    _msgCondVar.notify_all();
}

std::vector<MessageInfo> MessageManager::wait_message(int cur_rank, int streams_num) {
    std::vector<MessageInfo> messages_total;
    std::unique_lock<std::mutex> lock(_readMutex);
    _readCondVar.wait(lock, [&] {
        // std::cout << "wait_" << cur_rank << " : " << _readQueue[cur_rank].size() << " / " << streams_num << "\n";
        return _readQueue[cur_rank].size() >= streams_num;
    });
    std::swap(_readQueue[cur_rank], messages_total);
    // std::cout << "wait_" << cur_rank << " " << _readQueue[cur_rank].size() << " end\n";
    return messages_total;
}

void MessageManager::infer_wait() {
    std::unique_lock<std::mutex> lock(_inferMutex);
    _inferCondVar.wait(lock);
}

void MessageManager::reduce_wait(int cur_rank, int streams_num) {
    std::unique_lock<std::mutex> lock(_reduceMutex);
    while (_reduceQueue[cur_rank] < streams_num) {
        // std::cout << "reduce_wait_" << cur_rank << " " << _reduceQueue[cur_rank] << " end\n";
        _reduceCondVar.wait(lock);
    }
    _reduceQueue[cur_rank] = 0;
}

void MessageManager::server_wait(int streams_num) {
    if (!_serverThread.joinable()) {
        _readQueue.assign(streams_num, std::vector<MessageInfo>());
        _reduceQueue.assign(streams_num, 0);
        MsgType msg_type;
        _serverThread = std::thread([&, streams_num]() {
            int count = 0;
            int reduce_count = 0;
            while (!_isServerStopped) {
                std::vector<MessageInfo> msgQueue;
                {
                    // std::cout << "server_wait ........" << _isServerStopped << "\n";
                    std::unique_lock<std::mutex> lock(_msgMutex);
                    while (_messageQueue.empty()) {
                        _msgCondVar.wait(lock);
                    }
                    std::swap(_messageQueue, msgQueue);
                    // std::cout << "server_wait receive: " << msgQueue[0].msg_type << " rank:" << msgQueue[0].rank.size()
                    //           << " / " << msgQueue.size() << "\n";
                }

                for (auto rec_info : msgQueue) {
                    msg_type = rec_info.msg_type;
                    if (msg_type == START_INFER) {
                        Task task = std::move(rec_info.task);
                        task();
                    } else if (msg_type == TENSOR_PARALLEL) {
                        // Resend _readQueue that failed last time
                        bool stop = false;
                        while (!stop) {
                            stop = true;
                            for (int i = 0; i < streams_num; i++) {
                                if (_readQueue[i].size() == streams_num) {
                                    stop = false;
                                }
                            }
                            if (!stop) {
                                _readCondVar.notify_all();
                            }
                        }
                        for (int i = 0; i < streams_num; i++) {
                            std::lock_guard<std::mutex> lock(_readMutex);
                            _readQueue[i].push_back(rec_info);
                        }
                        _readCondVar.notify_all();
                    } else if (msg_type == CALL_BACK) {  // CALL_BACK
                        count++;
                        // std::cout << "server_wait CALL_BACK: " << count << "/" << streams_num << "\n";
                        if (count == streams_num) {
                            _inferCondVar.notify_one();
                            count = 0;
                        }
                    } else if (msg_type == REDUCE) {  // REDUCE
                        reduce_count++;
                        // std::cout << "server_wait REDUCE: " << reduce_count << "/" << streams_num << "\n";
                        if (reduce_count == streams_num) {
                            {
                                std::lock_guard<std::mutex> lock(_reduceMutex);
                                _reduceQueue.assign(streams_num, reduce_count);
                            }
                            _reduceCondVar.notify_all();
                            reduce_count = 0;
                        }
                    } else if (msg_type == QUIT) {
                        _isServerStopped = true;
                    }
                }
            }
            // std::cout << "-------- server_wait end ---------\n";
        });
    }
}

void MessageManager::set_sub_compiled_models(std::vector<std::shared_ptr<ov::ICompiledModel>> models) {
    _sub_compiled_models = models;
}

std::vector<std::shared_ptr<ov::ICompiledModel>> MessageManager::get_sub_compiled_models() {
    return _sub_compiled_models;
}

void MessageManager::set_sub_infer_requests(std::vector<std::shared_ptr<IAsyncInferRequest>> requests) {
    _sub_infer_requests = requests;
}

std::vector<std::shared_ptr<ov::IAsyncInferRequest>> MessageManager::get_sub_infer_requests() {
    return _sub_infer_requests;
}

void MessageManager::stop_server_thread() {
    MessageInfo msg_info;
    msg_info.msg_type = ov::threading::MsgType::QUIT;
    send_message(msg_info);
    if (_serverThread.joinable()) {
        _serverThread.join();
    }
}

void MessageManager::clear() {
    _sub_infer_requests.clear();
    _sub_compiled_models.clear();
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