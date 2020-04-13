// Copyright (C) 2018-2020 Intel Corporation
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
#include "ie_profiling.hpp"
#include "ie_parallel.hpp"
#include "ie_system_conf.h"
#include "ie_error.hpp"
#include "threading/ie_thread_affinity.hpp"
#include "details/ie_exception.hpp"
#include "ie_util_internal.hpp"
#include "threading/ie_cpu_streams_executor.hpp"

namespace InferenceEngine {
#if IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO
struct PinningObserver: public tbb::task_scheduler_observer {
    CpuSet& _mask;
    int     _ncpus                  = 0;
    int     _streamId               = 0;
    int     _threadsPerStream       = 0;
    int     _threadBindingStep      = 0;
    int     _threadBindingOffset    = 0;

    PinningObserver(tbb::task_arena&    arena,
                    CpuSet&             mask,
                    int                 ncpus,
                    const int           streamId,
                    const int           threadsPerStream,
                    const int           threadBindingStep,
                    const int           threadBindingOffset) :
        tbb::task_scheduler_observer(arena),
        _mask(mask),
        _ncpus(ncpus),
        _streamId(streamId),
        _threadsPerStream(threadsPerStream),
        _threadBindingStep(threadBindingStep),
        _threadBindingOffset(threadBindingOffset) {
        observe(true);
    }

    void on_scheduler_entry(bool) override {
        int threadIdx = tbb::task_arena::current_thread_index();
        int thrIdx = _streamId * _threadsPerStream + threadIdx + _threadBindingOffset;
        // pin thread to the vacant slot
        PinThreadToVacantCore(thrIdx, _threadBindingStep, _ncpus, _mask);
    }

    void on_scheduler_exit(bool) override {
        // reset the thread's mask (to the original process mask)
        PinCurrentThreadByMask(_ncpus, _mask);
    }

    ~PinningObserver() {
        observe(false);
    }
};
#endif  //  IE_THREAD != IE_THREAD_TBB

struct Stream {
    int _streamId   = 0;
    int _numaNodeId = 0;
#if IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO
    std::unique_ptr<tbb::task_arena> _taskArena;
    std::unique_ptr<PinningObserver> _pinningObserver;
#endif
};

struct CPUStreamsExecutor::Impl {
    std::string                 _name;
    std::vector<std::thread>    _threads;
    std::mutex                  _mutex;
    std::condition_variable     _queueCondVar;
    std::queue<Task>            _taskQueue;
    bool                        _isStopped = false;
    int                         _ncpus = 0;
    CpuSet                      _processMask;
    ThreadLocal<Stream*>        _localStream;
};

int CPUStreamsExecutor::GetStreamId() {
    auto stream = _impl->_localStream.local();
    if (nullptr == stream) THROW_IE_EXCEPTION << "Not in the stream thread";
    return stream->_streamId;
}

int CPUStreamsExecutor::GetNumaNodeId() {
    auto stream = _impl->_localStream.local();
    if (nullptr == stream) THROW_IE_EXCEPTION << "Not in the stream thread";
    return stream->_numaNodeId;
}

CPUStreamsExecutor::CPUStreamsExecutor(const IStreamsExecutor::Config& config) :
    _impl{new Impl} {
    IE_ASSERT(config._streams > 0);
    _impl->_name = config._name;
    auto numaNodes = getAvailableNUMANodes();
    IE_ASSERT(!numaNodes.empty());
    if (ThreadBindingType::CORES == config._threadBindingType) {
        std::tie(_impl->_processMask, _impl->_ncpus) = GetProcessMask();
    }
    for (auto streamId = 0; streamId < config._streams; ++streamId) {
        _impl->_threads.emplace_back([=] {
            annotateSetThreadName((_impl->_name + "_" + std::to_string(streamId)).c_str());
            Stream stream;
            stream._streamId   = streamId;
            stream._numaNodeId = numaNodes.at(streamId/((config._streams + numaNodes.size() - 1)/numaNodes.size()));
#if IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO
            auto concurrency = (0 == config._threadsPerStream) ? tbb::task_arena::automatic : config._threadsPerStream;
            if (ThreadBindingType::NUMA == config._threadBindingType) {
                stream._taskArena.reset(new tbb::task_arena(tbb::task_arena::constraints(stream._numaNodeId, concurrency)));
            } else if ((0 != config._threadsPerStream) || ThreadBindingType::CORES == config._threadBindingType) {
                stream._taskArena.reset(new tbb::task_arena(concurrency));
                if (ThreadBindingType::CORES == config._threadBindingType) {
                    if (nullptr != _impl->_processMask) {
                        stream._pinningObserver.reset(new PinningObserver{*stream._taskArena,
                                                                          _impl->_processMask,
                                                                          _impl->_ncpus,
                                                                          stream._streamId,
                                                                          config._threadsPerStream,
                                                                          config._threadBindingStep,
                                                                          config._threadBindingOffset});
                    }
                }
            }
#elif IE_THREAD == IE_THREAD_OMP
            omp_set_num_threads(config._threadsPerStream);
            if (!checkOpenMpEnvVars(false) && (ThreadBindingType::NONE != config._threadBindingType)) {
                if (nullptr != _impl->_processMask) {
                    parallel_nt(config._threadsPerStream, [&] (int threadIndex, int threadsPerStream) {
                        int thrIdx = stream._streamId * threadsPerStream + threadIndex + config._threadBindingOffset;
                        PinThreadToVacantCore(thrIdx, config._threadBindingStep, _impl->_ncpus, _impl->_processMask);
                    });
                }
            }
#elif IE_THREAD == IE_THREAD_SEQ
            if (ThreadBindingType::NUMA == config._threadBindingType) {
                PinCurrentThreadToSocket(stream._numaNodeId);
            } else if (ThreadBindingType::CORES == config._threadBindingType) {
                PinThreadToVacantCore(stream._streamId + config._threadBindingOffset, config._threadBindingStep, _impl->_ncpus, _impl->_processMask);
            }
#endif
            _impl->_localStream.local() = &stream;
            for (bool stopped = false; !stopped;) {
                Task currentTask;
                {  // waiting for the new task or for stop signal
                    std::unique_lock<std::mutex> lock(_impl->_mutex);
                    _impl->_queueCondVar.wait(lock, [&] { return !_impl->_taskQueue.empty() || (stopped = _impl->_isStopped); });
                    if (!_impl->_taskQueue.empty()) {
                        currentTask = std::move(_impl->_taskQueue.front());
                        _impl->_taskQueue.pop();
                    }
                }

                if (currentTask) {
#if IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO
                    if (nullptr != stream._taskArena) {
                        stream._taskArena->execute(std::move(currentTask));
                    } else {
                        currentTask();
                    }
#else
                    currentTask();
#endif
                }
            }
        });
    }
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

void CPUStreamsExecutor::run(Task task) {
    {
        std::lock_guard<std::mutex> lock(_impl->_mutex);
        _impl->_taskQueue.emplace(std::move(task));
    }
    _impl->_queueCondVar.notify_one();
}

}  // namespace InferenceEngine
