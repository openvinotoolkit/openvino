// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <queue>
#include <atomic>
#include <climits>
#include <cassert>
#include <utility>

#include "threading/ie_thread_local.hpp"
#include "ie_parallel.hpp"
#include "ie_system_conf.h"
#include "threading/ie_thread_affinity.hpp"
#include "details/ie_exception.hpp"
#include "threading/ie_cpu_streams_executor.hpp"
#include <openvino/itt.hpp>

using namespace openvino;

namespace InferenceEngine {
struct CPUStreamsExecutor::Impl {
    struct Stream {
#if IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO
        struct Observer: public tbb::task_scheduler_observer {
            CpuSet  _mask;
            int     _ncpus                  = 0;
            int     _threadBindingStep      = 0;
            int     _offset                 = 0;
            Observer(tbb::task_arena&    arena,
                     CpuSet              mask,
                     int                 ncpus,
                     const int           streamId,
                     const int           threadsPerStream,
                     const int           threadBindingStep,
                     const int           threadBindingOffset) :
                tbb::task_scheduler_observer(arena),
                _mask{std::move(mask)},
                _ncpus(ncpus),
                _threadBindingStep(threadBindingStep),
                _offset{streamId * threadsPerStream  + threadBindingOffset} {
            }
            void on_scheduler_entry(bool) override {
                PinThreadToVacantCore(_offset + tbb::this_task_arena::current_thread_index(), _threadBindingStep, _ncpus, _mask);
            }
            void on_scheduler_exit(bool) override {
                PinCurrentThreadByMask(_ncpus, _mask);
            }
            ~Observer() override = default;
        };
#endif
        explicit Stream(Impl* impl) :
            _impl(impl) {
            {
                std::lock_guard<std::mutex> lock{_impl->_streamIdMutex};
                if (_impl->_streamIdQueue.empty()) {
                    _streamId = _impl->_streamId++;
                } else {
                    _streamId = _impl->_streamIdQueue.front();
                    _impl->_streamIdQueue.pop();
                }
            }
#if IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO
            const auto concurrency = (0 == _impl->_config._threadsPerStream) ? tbb::task_arena::automatic : _impl->_config._threadsPerStream;
            #if TBB_INTERFACE_VERSION >= 12010 // TBB with hybrid CPU aware task_arena api
            if (ThreadBindingType::HYBRID_AWARE == _impl->_config._threadBindingType) {
               if (_impl->_config._threadUseBigCoresOnly) {
                    const auto selected_core_type = oneapi::tbb::info::core_types().back(); // runing on Big cores only
                    _taskArena.reset(new tbb::task_arena{ tbb::task_arena::constraints{selected_core_type, concurrency} });
                    // TODO: REMOVE THE DEBUG PRINTF
                    printf("%s, EXPLICIT BINDING, StreamId: %d (%d threads) assigned CORE TYPE : %d (CONCURRENCY: %d) \n",
                        _impl->_config._name.c_str(), _streamId, _impl->_config._threadsPerStream,
                        static_cast<int>(selected_core_type), concurrency);
                } else {
                    // assigning the stream to the core type in the round-robin fashion
                    // wrapping around total_streams (which is last value in the map) first
                    const int total_streams = _impl->total_streams_on_core_types.rbegin()->second;
                    const auto streamId_wrapped = _streamId % total_streams;
                    const auto& selected_core_type = std::find_if(_impl->total_streams_on_core_types.begin(), _impl->total_streams_on_core_types.end(),
                        [streamId_wrapped](const auto& p) { return p.second > streamId_wrapped; })->first;
                    _taskArena.reset(new tbb::task_arena{ tbb::task_arena::constraints{selected_core_type, concurrency} });
                    // TODO: REMOVE THE DEBUG PRINTF
                    printf("%s StreamId: %d (wrapped %d) assigned CORE TYPE : %d (total #streams: %d) \n",
                        _impl->_config._name.c_str(), _streamId, streamId_wrapped, static_cast<int>(selected_core_type), total_streams);
                }
            } else {
            #endif
                _numaNodeId = _impl->_config._streams
                    ? _impl->_usedNumaNodes.at(
                        (_streamId % _impl->_config._streams)/
                        ((_impl->_config._streams + _impl->_usedNumaNodes.size() - 1)/_impl->_usedNumaNodes.size()))
                    : _impl->_usedNumaNodes.at(_streamId % _impl->_usedNumaNodes.size());
                if (ThreadBindingType::NUMA == _impl->_config._threadBindingType) {
                    // TODO: REMOVE THE DEBUG PRINTF
                    printf("%s, conventional ThreadBindingType::NUMA codepath \n", _impl->_config._name.c_str());
                    #if TBB_INTERFACE_VERSION >= 11100  // TBB has numa aware task_arena api
                    _taskArena.reset(new tbb::task_arena{tbb::task_arena::constraints{_numaNodeId, concurrency}});
                    #else
                    _taskArena.reset(new tbb::task_arena{concurrency});
                    #endif
                } else if ((0 != _impl->_config._threadsPerStream) || (ThreadBindingType::CORES == _impl->_config._threadBindingType)) {
                    // TODO: REMOVE THE DEBUG PRINTF
                    printf("%s, conventional ThreadBindingType::CORES codepath \n", _impl->_config._name.c_str());
                    _taskArena.reset(new tbb::task_arena{concurrency});
                    if (ThreadBindingType::CORES == _impl->_config._threadBindingType) {
                        CpuSet processMask;
                        int    ncpus = 0;
                        std::tie(processMask, ncpus) = GetProcessMask();
                        if (nullptr != processMask) {
                            _observer.reset(new Observer{*_taskArena,
                                                         std::move(processMask),
                                                         ncpus,
                                                         _streamId,
                                                         _impl->_config._threadsPerStream,
                                                         _impl->_config._threadBindingStep,
                                                         _impl->_config._threadBindingOffset});
                            _observer->observe(true);
                        }
                    }
                }
            #if TBB_INTERFACE_VERSION >= 12010 // TBB with hybrid CPU aware task_arena api
            } // closing the if-else-clause on the hybrid cores
            #endif
#elif IE_THREAD == IE_THREAD_OMP
            omp_set_num_threads(_impl->_config._threadsPerStream);
            if (!checkOpenMpEnvVars(false) && (ThreadBindingType::NONE != _impl->_config._threadBindingType)) {
                CpuSet processMask;
                int    ncpus = 0;
                std::tie(processMask, ncpus) = GetProcessMask();
                if (nullptr != processMask) {
                    parallel_nt(_impl->_config._threadsPerStream, [&] (int threadIndex, int threadsPerStream) {
                        int thrIdx = _streamId * _impl->_config._threadsPerStream + threadIndex + _impl->_config._threadBindingOffset;
                        PinThreadToVacantCore(thrIdx, _impl->_config._threadBindingStep, ncpus, processMask);
                    });
                }
            }
#elif IE_THREAD == IE_THREAD_SEQ
            if (ThreadBindingType::NUMA == _impl->_config._threadBindingType) {
                PinCurrentThreadToSocket(_numaNodeId);
            } else if (ThreadBindingType::CORES == _impl->_config._threadBindingType) {
                CpuSet processMask;
                int    ncpus = 0;
                std::tie(processMask, ncpus) = GetProcessMask();
                if (nullptr != processMask) {
                    PinThreadToVacantCore(_streamId + _impl->_config._threadBindingOffset, _impl->_config._threadBindingStep, ncpus, processMask);
                }
            }
#endif
        }
        ~Stream() {
            {
                std::lock_guard<std::mutex> lock{_impl->_streamIdMutex};
                _impl->_streamIdQueue.push(_streamId);
            }
#if IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO
            if (nullptr != _observer) {
                _observer->observe(false);
            }
#endif
        }

        Impl* _impl     = nullptr;
        int _streamId   = 0;
        int _numaNodeId = 0;
        bool _execute = false;
        std::queue<Task> _taskQueue;
#if IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO
        std::unique_ptr<tbb::task_arena>    _taskArena;
        std::unique_ptr<Observer>           _observer;
#endif
    };

    explicit Impl(const Config& config) :
        _config{config},
        _streams([this] {
            return std::make_shared<Impl::Stream>(this);
        }) {
        // TODO: REMOVE THE DEBUG PRINTF
        printf("INIT (%s), STREAMS: %d, THREADS_PER_STREAM: %d \n", _config._name.c_str(), _config._streams, _config._threadsPerStream);
        auto numaNodes = getAvailableNUMANodes();
        if (_config._streams != 0) {
            std::copy_n(std::begin(numaNodes),
                        std::min(static_cast<std::size_t>(_config._streams), numaNodes.size()),
                        std::back_inserter(_usedNumaNodes));
        } else {
            _usedNumaNodes = numaNodes;
        }

        #if defined(TBB_INTERFACE_VERSION) && (TBB_INTERFACE_VERSION >= 12010) // TBB with hybrid CPU aware task_arena api
        const auto core_types = oneapi::tbb::info::core_types();
        const int threadsPerStream = (0 == config._threadsPerStream) ? std::thread::hardware_concurrency() : config._threadsPerStream;
        int sum = 0;
        for (auto iter = core_types.begin(); iter < core_types.end(); iter++) {
            const auto& type = *iter;
            // calculating the #streams per core type
            const int num_streams_for_core_type = std::max(1, oneapi::tbb::info::default_concurrency(type) / threadsPerStream);
            sum += num_streams_for_core_type;
            // prefix sum, so the core type for a given stream id will be deduced just as a upper_bound
            // (notice that the map keeps the elements in the descending order, so the big cores are populated first)
            total_streams_on_core_types[type] = sum;
        }
        // TODO: REMOVE THE DEBUG PRINTF
        printf("%s total_streams_on_core_types: [%d core_type, %d #streams], [%d core_type, %d #streams]) \n",
            _config._name.c_str(), static_cast<int>(total_streams_on_core_types.begin()->first), total_streams_on_core_types.begin()->second,
            static_cast<int>(total_streams_on_core_types.rbegin()->first), total_streams_on_core_types.rbegin()->second);
        #endif
        for (auto streamId = 0; streamId < _config._streams; ++streamId) {
            _threads.emplace_back([this, streamId] {
                openvino::itt::threadName(_config._name + "_" + std::to_string(streamId));
                for (bool stopped = false; !stopped;) {
                    Task task;
                    {
                        std::unique_lock<std::mutex> lock(_mutex);
                        _queueCondVar.wait(lock, [&] { return !_taskQueue.empty() || (stopped = _isStopped); });
                        if (!_taskQueue.empty()) {
                            task = std::move(_taskQueue.front());
                            _taskQueue.pop();
                        }
                    }
                    if (task) {
                        Execute(task, *(_streams.local()));
                    }
                }
            });
        }
    }

    void Enqueue(Task task) {
        {
            std::lock_guard<std::mutex> lock(_mutex);
            _taskQueue.emplace(std::move(task));
        }
        _queueCondVar.notify_one();
    }

    void Execute(const Task& task, Stream& stream) {
#if IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO
        auto& arena = stream._taskArena;
        if (nullptr != arena) {
            arena->execute(std::move(task));
        } else {
            task();
        }
#else
        task();
#endif
    }

    void Defer(Task task) {
        auto& stream = *(_streams.local());
        stream._taskQueue.push(std::move(task));
        if (!stream._execute) {
            stream._execute = true;
            try {
                while (!stream._taskQueue.empty()) {
                    Execute(stream._taskQueue.front(), stream);
                    stream._taskQueue.pop();
                }
            } catch(...) {}
            stream._execute = false;
        }
    }

    Config                                  _config;
    std::mutex                              _streamIdMutex;
    int                                     _streamId = 0;
    std::queue<int>                         _streamIdQueue;
    std::vector<std::thread>                _threads;
    std::mutex                              _mutex;
    std::condition_variable                 _queueCondVar;
    std::queue<Task>                        _taskQueue;
    bool                                    _isStopped = false;
    std::vector<int>                        _usedNumaNodes;
    ThreadLocal<std::shared_ptr<Stream>>    _streams;
    #if TBB_INTERFACE_VERSION >= 12010 // TBB with hybrid CPU aware task_arena api
    // streams mapping to the core types
    // stored in the reversed order (so the big cores, with the highest core_type_id value, are populated first)
    using MapStreamIdToCoreTypes = std::map<oneapi::tbb::core_type_id, int, std::greater<int>>;
    MapStreamIdToCoreTypes total_streams_on_core_types;
    #endif
};


int CPUStreamsExecutor::GetStreamId() {
    auto stream = _impl->_streams.local();
    return stream->_streamId;
}

int CPUStreamsExecutor::GetNumaNodeId() {
    auto stream = _impl->_streams.local();
    return stream->_numaNodeId;
}

CPUStreamsExecutor::CPUStreamsExecutor(const IStreamsExecutor::Config& config) :
    _impl{new Impl{config}} {
}

CPUStreamsExecutor::~CPUStreamsExecutor() {
    {
        std::lock_guard<std::mutex> lock(_impl->_mutex);
        _impl->_isStopped = true;
    }
    _impl->_queueCondVar.notify_all();
    for (auto& thread : _impl->_threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void CPUStreamsExecutor::Execute(Task task) {
    _impl->Defer(std::move(task));
}

void CPUStreamsExecutor::run(Task task) {
    if (0 == _impl->_config._streams) {
        _impl->Defer(std::move(task));
    } else {
        _impl->Enqueue(std::move(task));
    }
}

}  // namespace InferenceEngine
