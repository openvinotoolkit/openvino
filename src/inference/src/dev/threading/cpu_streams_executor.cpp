// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/threading/cpu_streams_executor.hpp"

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <set>
#include <thread>
#include <vector>

#include "dev/threading/parallel_custom_arena.hpp"
#include "dev/threading/thread_affinity.hpp"
#include "openvino/itt.hpp"
#include "openvino/runtime/system_conf.hpp"
#include "openvino/runtime/threading/cpu_streams_executor_internal.hpp"
#include "openvino/runtime/threading/cpu_streams_info.hpp"
#include "openvino/runtime/threading/executor_manager.hpp"
#include "openvino/runtime/threading/thread_local.hpp"

namespace ov {
namespace threading {
struct CPUStreamsExecutor::Impl {
    struct Stream {
#if OV_THREAD == OV_THREAD_TBB || OV_THREAD == OV_THREAD_TBB_AUTO
        struct Observer : public custom::task_scheduler_observer {
            CpuSet _mask;
            int _ncpus = 0;
            int _threadBindingStep = 0;
            std::vector<int> _cpu_ids;
            Observer(custom::task_arena& arena, CpuSet mask, int ncpus, const std::vector<int> cpu_ids = {})
                : custom::task_scheduler_observer(arena),
                  _mask{std::move(mask)},
                  _ncpus(ncpus),
                  _cpu_ids(cpu_ids) {}
            void on_scheduler_entry(bool) override {
                pin_thread_to_vacant_core(tbb::this_task_arena::current_thread_index(),
                                          _threadBindingStep,
                                          _ncpus,
                                          _mask,
                                          _cpu_ids);
            }
            void on_scheduler_exit(bool) override {
                pin_current_thread_by_mask(_ncpus, _mask);
            }
            ~Observer() override = default;
        };
#endif
        explicit Stream(Impl* impl) : _impl(impl) {
            {
                std::lock_guard<std::mutex> lock{_impl->_streamIdMutex};
                if (_impl->_streamIdQueue.empty()) {
                    _streamId = _impl->_streamId++;
                } else {
                    _streamId = _impl->_streamIdQueue.front();
                    _impl->_streamIdQueue.pop();
                }
            }
            _numaNodeId =
                _impl->_config.get_streams()
                    ? _impl->_usedNumaNodes.at((_streamId % _impl->_config.get_streams()) /
                                               ((_impl->_config.get_streams() + _impl->_usedNumaNodes.size() - 1) /
                                                _impl->_usedNumaNodes.size()))
                    : _impl->_usedNumaNodes.at(_streamId % _impl->_usedNumaNodes.size());
#if OV_THREAD == OV_THREAD_TBB || OV_THREAD == OV_THREAD_TBB_AUTO
            if (_impl->_config.get_streams_info_table().size() > 0) {
                init_stream();
            }
#elif OV_THREAD == OV_THREAD_OMP
            omp_set_num_threads(_impl->_config.get_threads_per_stream());
            if (!check_open_mp_env_vars(false) && _impl->_config.get_cpu_pinning()) {
                CpuSet processMask;
                int ncpus = 0;
                std::tie(processMask, ncpus) = get_process_mask();
                if (nullptr != processMask) {
                    parallel_nt(_impl->_config.get_threads_per_stream(), [&](int threadIndex, int threadsPerStream) {
                        int thrIdx = _streamId * _impl->_config.get_threads_per_stream() + threadIndex +
                                     _impl->_config.get_thread_binding_offset();
                        pin_thread_to_vacant_core(thrIdx, _impl->_config.get_thread_binding_step(), ncpus, processMask);
                    });
                }
            }
#endif
        }
        ~Stream() {
            {
                std::lock_guard<std::mutex> lock{_impl->_streamIdMutex};
                _impl->_streamIdQueue.push(_streamId);
            }
#if OV_THREAD == OV_THREAD_TBB || OV_THREAD == OV_THREAD_TBB_AUTO
            if (nullptr != _observer) {
                _observer->observe(false);
            }
#endif
        }

#if OV_THREAD == OV_THREAD_TBB || OV_THREAD == OV_THREAD_TBB_AUTO
        void create_tbb_task_arena(const int stream_id,
                                   const StreamCreateType stream_type,
                                   const int concurrency,
                                   const int core_type,
                                   const int numa_node_id,
                                   const int socket_id,
                                   const int max_threads_per_core) {
            auto stream_processors = _impl->_config.get_stream_processor_ids();
            _numaNodeId = numa_node_id;
            _socketId = socket_id;
            if (stream_type == STREAM_WITHOUT_PARAM) {
                _taskArena.reset(new custom::task_arena{custom::task_arena::constraints{}
                                                            .set_max_concurrency(concurrency)
                                                            .set_max_threads_per_core(max_threads_per_core)});
            } else if (stream_type == STREAM_WITH_NUMA_ID) {
                // Numa node id has used different mapping methods in TBBBind since oneTBB 2021.4.0
#    if USE_TBBBIND_2_5
                auto real_numa_node_id = _numaNodeId;
#    else
                auto real_numa_node_id = get_org_numa_id(_numaNodeId);
                int tbb_version;
#        if (TBB_INTERFACE_VERSION < 12000)
                tbb_version = tbb::TBB_runtime_interface_version();
#        else
                tbb_version = TBB_runtime_interface_version();
#        endif
                if (tbb_version >= 12040) {
                    real_numa_node_id = _numaNodeId;
                }
#    endif
                _taskArena.reset(new custom::task_arena{custom::task_arena::constraints{}
                                                            .set_numa_id(real_numa_node_id)
                                                            .set_max_concurrency(concurrency)
                                                            .set_max_threads_per_core(max_threads_per_core)});
            } else if (stream_type == STREAM_WITH_CORE_TYPE) {
                const auto real_core_type = (core_type == MAIN_CORE_PROC || core_type == HYPER_THREADING_PROC)
                                                ? custom::info::core_types().back()
                                                : custom::info::core_types().front();
                _taskArena.reset(new custom::task_arena{custom::task_arena::constraints{}
                                                            .set_core_type(real_core_type)
                                                            .set_max_concurrency(concurrency)
                                                            .set_max_threads_per_core(max_threads_per_core)});
            } else {
                _taskArena.reset(new custom::task_arena{concurrency});
                _cpu_ids =
                    stream_id < static_cast<int>(stream_processors.size()) ? stream_processors[stream_id] : _cpu_ids;
                if (_cpu_ids.size() > 0) {
                    CpuSet processMask;
                    int ncpus = 0;
                    std::tie(processMask, ncpus) = get_process_mask();
                    if (nullptr != processMask) {
                        _observer.reset(new Observer{*_taskArena, std::move(processMask), ncpus, _cpu_ids});
                        _observer->observe(true);
                    }
                }
            }
        }
        void init_stream() {
            int concurrency;
            int cpu_core_type;
            int numa_node_id;
            int socket_id;
            int max_threads_per_core;
            StreamCreateType stream_type;
            const auto org_proc_type_table = get_org_proc_type_table();
            int streams_num = _impl->_config.get_streams();
            const auto stream_id = streams_num == 0 ? 0 : _streamId % streams_num;
            _rank = _impl->_config.get_rank();
            get_cur_stream_info(stream_id,
                                _impl->_config.get_cpu_pinning(),
                                org_proc_type_table,
                                _impl->_config.get_streams_info_table(),
                                stream_type,
                                concurrency,
                                cpu_core_type,
                                numa_node_id,
                                socket_id,
                                max_threads_per_core);
            if (concurrency <= 0) {
                return;
            }
            create_tbb_task_arena(stream_id,
                                  stream_type,
                                  concurrency,
                                  cpu_core_type,
                                  numa_node_id,
                                  socket_id,
                                  max_threads_per_core);
        }
#endif

        Impl* _impl = nullptr;
        int _streamId = 0;
        int _numaNodeId = 0;
        int _socketId = 0;
        bool _execute = false;
        std::vector<int> _rank;
        std::queue<Task> _taskQueue;
#if OV_THREAD == OV_THREAD_TBB || OV_THREAD == OV_THREAD_TBB_AUTO
        std::unique_ptr<custom::task_arena> _taskArena;
        std::unique_ptr<Observer> _observer;
        std::vector<int> _cpu_ids;
#elif OV_THREAD == OV_THREAD_SEQ
        CpuSet _mask = nullptr;
        int _ncpus = 0;
#endif
    };
    // if the thread is created by CPUStreamsExecutor, the Impl::Stream of the thread is stored by tbb Class
    // enumerable_thread_specific, the alias is ThreadLocal, the limitations of ThreadLocal please refer to
    // https://spec.oneapi.io/versions/latest/elements/oneTBB/source/thread_local_storage/enumerable_thread_specific_cls.html
    // if the thread is created by customer, the Impl::Stream of the thread will be stored in variable _stream_map, and
    // will be counted by thread_local t_stream_count_map.
    // when the customer's thread is destoryed, the stream's count will became 1,
    // Call local() will reuse one of them, and release others.
    // it's only a workaround for ticket CVS-111490, please be carefully when need to modify
    // CustomeThreadLocal::local(), especially like operations that will affect the count of
    // CustomThreadLocal::ThreadId
    class CustomThreadLocal : public ThreadLocal<std::shared_ptr<Stream>> {
        class ThreadTracker {
        public:
            explicit ThreadTracker(const std::thread::id& id)
                : _id(id),
                  _count_ptr(std::make_shared<std::atomic_int>(1)) {}
            ~ThreadTracker() {
                _count_ptr->fetch_sub(1);
            }
            std::shared_ptr<ThreadTracker> fetch() {
                auto new_ptr = std::shared_ptr<ThreadTracker>(new ThreadTracker(*this));
                auto pre_valule = new_ptr.get()->_count_ptr->fetch_add(1);
                OPENVINO_ASSERT(pre_valule == 1, "this value must be 1, please check code CustomThreadLocal::local()");
                return new_ptr;
            }
            const std::thread::id& get_id() const {
                return _id;
            }
            int count() const {
                return *(_count_ptr.get());
            }

        private:
            // disable all copy and move semantics, user only can use fetch()
            // to create a new instance with a shared count num;
            ThreadTracker(ThreadTracker const&) = default;
            ThreadTracker(ThreadTracker&&) = delete;
            ThreadTracker& operator=(ThreadTracker const&) = delete;
            ThreadTracker& operator=(ThreadTracker&&) = delete;
            std::thread::id _id;
            std::shared_ptr<std::atomic_int> _count_ptr;
        };

    public:
        CustomThreadLocal(std::function<std::shared_ptr<Stream>()> callback_construct, Impl* impl)
            : ThreadLocal<std::shared_ptr<Stream>>(std::move(callback_construct)),
              _impl(impl) {}
        std::shared_ptr<Stream> local() {
            // maybe there are two CPUStreamsExecutors in the same thread.
            static thread_local std::map<void*, std::shared_ptr<CustomThreadLocal::ThreadTracker>> t_stream_count_map;
            // fix the memory leak issue that CPUStreamsExecutor is already released,
            // but still exists CustomThreadLocal::ThreadTracker in t_stream_count_map
            for (auto it = t_stream_count_map.begin(); it != t_stream_count_map.end();) {
                if (this != it->first && it->second->count() == 1) {
                    t_stream_count_map.erase(it++);
                } else {
                    it++;
                }
            }
            auto id = std::this_thread::get_id();
            auto search = _thread_ids.find(id);
            if (search != _thread_ids.end()) {
                return ThreadLocal<std::shared_ptr<Stream>>::local();
            }
            std::lock_guard<std::mutex> guard(_stream_map_mutex);
            for (auto& item : _stream_map) {
                if (item.first->get_id() == id) {
                    // check if the ThreadTracker of this stream is already in t_stream_count_map
                    // if not, then create ThreadTracker for it
                    auto iter = t_stream_count_map.find((void*)this);
                    if (iter == t_stream_count_map.end()) {
                        t_stream_count_map[(void*)this] = item.first->fetch();
                    }
                    return item.second;
                }
            }
            std::shared_ptr<Impl::Stream> stream = nullptr;
            for (auto it = _stream_map.begin(); it != _stream_map.end();) {
                if (it->first->count() == 1) {
                    if (stream == nullptr) {
                        stream = it->second;
                    }
                    _stream_map.erase(it++);
                } else {
                    it++;
                }
            }
            if (stream == nullptr) {
                stream = std::make_shared<Impl::Stream>(_impl);
            }
            auto tracker_ptr = std::make_shared<CustomThreadLocal::ThreadTracker>(id);
            t_stream_count_map[(void*)this] = tracker_ptr;
            auto new_tracker_ptr = tracker_ptr->fetch();
            _stream_map[new_tracker_ptr] = stream;
            return stream;
        }

        void set_thread_ids_map(std::vector<std::thread>& threads) {
            for (auto& thread : threads) {
                _thread_ids.insert(thread.get_id());
            }
        }

        bool find_thread_id() {
            auto id = std::this_thread::get_id();
            auto search = _thread_ids.find(id);
            if (search != _thread_ids.end()) {
                return true;
            }
            std::lock_guard<std::mutex> guard(_stream_map_mutex);
            for (auto& item : _stream_map) {
                if (item.first->get_id() == id) {
                    return true;
                }
            }
            return false;
        }

    private:
        std::set<std::thread::id> _thread_ids;
        Impl* _impl;
        std::map<std::shared_ptr<CustomThreadLocal::ThreadTracker>, std::shared_ptr<Impl::Stream>> _stream_map;
        std::mutex _stream_map_mutex;
    };

    explicit Impl(const Config& config)
        : _config{config},
          _streams(
              [this] {
                  return std::make_shared<Impl::Stream>(this);
              },
              this) {
        _exectorMgr = executor_manager();
        auto numaNodes = get_available_numa_nodes();
        int streams_num = _config.get_streams();
        auto processor_ids = _config.get_stream_processor_ids();
        if (streams_num != 0) {
            std::copy_n(std::begin(numaNodes),
                        std::min<std::size_t>(streams_num, numaNodes.size()),
                        std::back_inserter(_usedNumaNodes));
        } else {
            _usedNumaNodes = std::move(numaNodes);
        }
        for (auto streamId = 0; streamId < streams_num; ++streamId) {
            if (_config.get_cpu_reservation()) {
                std::lock_guard<std::mutex> lock(_cpu_ids_mutex);
                _cpu_ids_all.insert(_cpu_ids_all.end(), processor_ids[streamId].begin(), processor_ids[streamId].end());
            }
            _threads.emplace_back([this, streamId] {
                openvino::itt::threadName(_config.get_name() + "_" + std::to_string(streamId));
                for (bool stopped = false; !stopped;) {
                    Task task;
                    {
                        std::unique_lock<std::mutex> lock(_mutex);
                        _queueCondVar.wait(lock, [&] {
                            return !_taskQueue.empty() || (stopped = _isStopped);
                        });
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
        _streams.set_thread_ids_map(_threads);
    }

    void Enqueue(Task task) {
        {
            std::lock_guard<std::mutex> lock(_mutex);
            _taskQueue.emplace(std::move(task));
        }
        _queueCondVar.notify_one();
    }

    void Execute(const Task& task, Stream& stream) {
#if OV_THREAD == OV_THREAD_TBB || OV_THREAD == OV_THREAD_TBB_AUTO
        auto& arena = stream._taskArena;
        if (nullptr != arena) {
            arena->execute(std::move(task));
        } else {
            task();
        }
#else
        pin_stream_to_cpus();
        task();
        unpin_stream_to_cpus();
#endif
    }

    void pin_stream_to_cpus() {
#if OV_THREAD == OV_THREAD_SEQ
        if (_config.get_cpu_pinning()) {
            auto stream = _streams.local();
            auto proc_type_table = get_org_proc_type_table();
            std::tie(stream->_mask, stream->_ncpus) = get_process_mask();
            if (get_num_numa_nodes() > 1) {
                pin_current_thread_to_socket(stream->_numaNodeId);
            } else if (proc_type_table.size() == 1 && proc_type_table[0][EFFICIENT_CORE_PROC] == 0) {
                if (nullptr != stream->_mask) {
                    pin_thread_to_vacant_core(stream->_streamId + _config.get_thread_binding_offset(),
                                              _config.get_thread_binding_step(),
                                              stream->_ncpus,
                                              stream->_mask);
                }
            }
        }
#endif
    }

    void unpin_stream_to_cpus() {
#if OV_THREAD == OV_THREAD_SEQ
        auto stream = _streams.local();
        if (stream->_mask) {
            pin_current_thread_by_mask(stream->_ncpus, stream->_mask);
        }
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
            } catch (...) {
            }
            stream._execute = false;
        }
    }

    Config _config;
    std::mutex _streamIdMutex;
    int _streamId = 0;
    std::queue<int> _streamIdQueue;
    std::vector<std::thread> _threads;
    std::mutex _mutex;
    std::condition_variable _queueCondVar;
    std::queue<Task> _taskQueue;
    bool _isStopped = false;
    std::vector<int> _usedNumaNodes;
    CustomThreadLocal _streams;
    std::shared_ptr<ExecutorManager> _exectorMgr;
    bool _isExit = false;
    std::vector<int> _cpu_ids_all;
    std::mutex _cpu_ids_mutex;
};

int CPUStreamsExecutor::get_stream_id() {
    if (!_impl->_streams.find_thread_id()) {
        return 0;
    }
    auto stream = _impl->_streams.local();
    return stream->_streamId;
}

int CPUStreamsExecutor::get_streams_num() {
    return _impl->_config.get_streams();
}

int CPUStreamsExecutor::get_numa_node_id() {
    if (!_impl->_streams.find_thread_id()) {
        return 0;
    }
    auto stream = _impl->_streams.local();
    return stream->_numaNodeId;
}

int CPUStreamsExecutor::get_socket_id() {
    if (!_impl->_streams.find_thread_id()) {
        return 0;
    }
    auto stream = _impl->_streams.local();
    return stream->_socketId;
}

std::vector<int> CPUStreamsExecutor::get_rank() {
    auto stream = _impl->_streams.local();
    return stream->_rank;
}

void CPUStreamsExecutor::cpu_reset() {
    {
        std::lock_guard<std::mutex> lock(_impl->_cpu_ids_mutex);
        if (!_impl->_cpu_ids_all.empty()) {
            set_cpu_used(_impl->_cpu_ids_all, NOT_USED);
            _impl->_cpu_ids_all.clear();
        }
    }
}

CPUStreamsExecutor::CPUStreamsExecutor(const IStreamsExecutor::Config& config) : _impl{new Impl{config}} {}

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

void CPUStreamsExecutor::execute(Task task) {
    _impl->Defer(std::move(task));
}

void CPUStreamsExecutor::run(Task task) {
    if (0 == _impl->_config.get_streams()) {
        _impl->Defer(std::move(task));
    } else {
        _impl->Enqueue(std::move(task));
    }
}

}  // namespace threading
}  // namespace ov
