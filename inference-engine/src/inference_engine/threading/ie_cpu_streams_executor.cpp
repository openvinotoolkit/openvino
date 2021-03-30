// Copyright (C) 2018-2021 Intel Corporation
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
#include <iostream>

#include "threading/ie_thread_local.hpp"
#include "ie_parallel.hpp"
#include "ie_system_conf.h"
#include "threading/ie_thread_affinity.hpp"
#include "threading/ie_cpu_streams_executor.hpp"
#include <openvino/itt.hpp>

using namespace openvino;

#if IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO
namespace custom {
namespace detail {

extern "C" {
void __TBB_internal_initialize_system_topology(
    std::size_t groups_num,
    int& numa_nodes_count, int*& numa_indexes_list,
    int& core_types_count, int*& core_types_indexes_list
);
binding_handler* __TBB_internal_allocate_binding_handler(int number_of_slots, int numa_id, int core_type_id, int max_threads_per_core);
void __TBB_internal_deallocate_binding_handler(binding_handler* handler_ptr);
void __TBB_internal_apply_affinity(binding_handler* handler_ptr, int slot_num);
void __TBB_internal_restore_affinity(binding_handler* handler_ptr, int slot_num);
int __TBB_internal_get_default_concurrency(int numa_id, int core_type_id, int max_threads_per_core);
}

int get_processors_group_num() {
#if defined(_WIN32) || defined(_WIN64)
    SYSTEM_INFO si;
    GetNativeSystemInfo(&si);

    DWORD_PTR pam, sam, m = 1;
    GetProcessAffinityMask(GetCurrentProcess(), &pam, &sam);
    int nproc = 0;
    for (std::size_t i = 0; i < sizeof(DWORD_PTR) * CHAR_BIT; ++i, m <<= 1) {
        if ( pam & m )
            ++nproc;
    }
    if (nproc == static_cast<int>(si.dwNumberOfProcessors)) {
        return GetActiveProcessorGroupCount();
    }
#endif
    return 1;
}

bool is_binding_environment_valid() {
#if defined(_WIN32) && !defined(_WIN64)
    static bool result = [] {
        // For 32-bit Windows applications, process affinity masks can only support up to 32 logical CPUs.
        SYSTEM_INFO si;
        GetNativeSystemInfo(&si);
        if (si.dwNumberOfProcessors > 32) return false;
        return true;
    }();
    return result;
#else
    return true;
#endif /* _WIN32 && !_WIN64 */
}

int  numa_nodes_count = 0;
int* numa_nodes_indexes = nullptr;

int  core_types_count = 0;
int* core_types_indexes = nullptr;

void initialize_system_topology() {
    static std::once_flag is_topology_initialized;

    std::call_once(is_topology_initialized, [&]{
        if (is_binding_environment_valid()) {
            __TBB_internal_initialize_system_topology(
                get_processors_group_num(),
                numa_nodes_count, numa_nodes_indexes,
                core_types_count, core_types_indexes);
        } else {
            static int dummy_index = tbb::task_arena::automatic;

            numa_nodes_count = 1;
            numa_nodes_indexes = &dummy_index;

            core_types_count = 1;
            core_types_indexes = &dummy_index;
        }
    });
}
} // namespace detail

binding_observer::binding_observer(tbb::task_arena& ta, int num_slots, int numa_id, core_type_id core_type, int max_threads_per_core)
    : task_scheduler_observer(ta) {
    detail::initialize_system_topology();
    my_binding_handler = detail::__TBB_internal_allocate_binding_handler(num_slots, numa_id, core_type, max_threads_per_core);
}

void binding_observer::on_scheduler_entry(bool) {
    detail::__TBB_internal_apply_affinity(my_binding_handler, tbb::this_task_arena::current_thread_index());
}

void binding_observer::on_scheduler_exit(bool) {
    detail::__TBB_internal_restore_affinity(my_binding_handler, tbb::this_task_arena::current_thread_index());
}

binding_observer::~binding_observer() {
    detail::__TBB_internal_deallocate_binding_handler(my_binding_handler);
}

binding_observer* construct_binding_observer(tbb::task_arena& ta, int num_slots,
                                             numa_node_id numa_id = tbb::task_arena::automatic,
                                             core_type_id core_type = tbb::task_arena::automatic,
                                             int max_threads_per_core = tbb::task_arena::automatic) {
    binding_observer* observer = nullptr;
    if (detail::is_binding_environment_valid() &&
      ((core_type >= 0 && info::core_types().size() > 1) || (numa_id >= 0 && info::numa_nodes().size() > 1) || max_threads_per_core > 0)) {
        observer = new binding_observer(ta, num_slots, numa_id, core_type, max_threads_per_core);
        observer->observe(true);
    }
    return observer;
}

namespace info {
std::vector<numa_node_id> numa_nodes() {
    detail::initialize_system_topology();
    std::vector<numa_node_id> node_indices(detail::numa_nodes_count);
    std::memcpy(node_indices.data(), detail::numa_nodes_indexes, detail::numa_nodes_count * sizeof(int));
    return node_indices;
}

std::vector<core_type_id> core_types() {
    detail::initialize_system_topology();
    std::vector<numa_node_id> core_type_indexes(detail::core_types_count);
    std::memcpy(core_type_indexes.data(), detail::core_types_indexes, detail::core_types_count * sizeof(int));
    return core_type_indexes;
}

int default_concurrency(numa_node_id numa_id, core_type_id core_type_id, int max_threads_per_core) {
    if (detail::is_binding_environment_valid()) {
        detail::initialize_system_topology();
        return detail::__TBB_internal_get_default_concurrency(numa_id, core_type_id, max_threads_per_core);
    }
    return tbb::this_task_arena::max_concurrency();
}
} // namespace info
} // namespace custom

#endif /*IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO*/

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
            _numaNodeId = _impl->_config._streams
                ? _impl->_usedNumaNodes.at(
                    (_streamId % _impl->_config._streams)/
                    ((_impl->_config._streams + _impl->_usedNumaNodes.size() - 1)/_impl->_usedNumaNodes.size()))
                : _impl->_usedNumaNodes.at(_streamId % _impl->_usedNumaNodes.size());
#if IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO
            auto latency_core_type = custom::info::core_types().back();
            _taskArena.reset(new tbb::task_arena{
                custom::info::default_concurrency(tbb::task_arena::automatic, latency_core_type)});
            _observer.reset(custom::construct_binding_observer(
                *_taskArena,
                _taskArena->max_concurrency(),
                /*numa_id*/tbb::task_arena::automatic,
                /*core_type_id*/latency_core_type));
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
        std::unique_ptr<tbb::task_arena>          _taskArena;
        std::unique_ptr<custom::binding_observer> _observer;
#endif
    };

    explicit Impl(const Config& config) :
        _config{config},
        _streams([this] {
            return std::make_shared<Impl::Stream>(this);
        }) {
        auto numaNodes = getAvailableNUMANodes();
        if (_config._streams != 0) {
            std::copy_n(std::begin(numaNodes),
                        std::min(static_cast<std::size_t>(_config._streams), numaNodes.size()),
                        std::back_inserter(_usedNumaNodes));
        } else {
            _usedNumaNodes = numaNodes;
        }
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
