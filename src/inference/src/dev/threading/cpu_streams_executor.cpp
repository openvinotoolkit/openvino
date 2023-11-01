// Copyright (C) 2018-2023 Intel Corporation
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
            int _offset = 0;
            int _cpuIdxOffset = 0;
            std::vector<int> _cpu_ids;
            Observer(custom::task_arena& arena,
                     CpuSet mask,
                     int ncpus,
                     const int streamId,
                     const int threadsPerStream,
                     const int threadBindingStep,
                     const int threadBindingOffset,
                     const int cpuIdxOffset = 0,
                     const std::vector<int> cpu_ids = {})
                : custom::task_scheduler_observer(arena),
                  _mask{std::move(mask)},
                  _ncpus(ncpus),
                  _threadBindingStep(threadBindingStep),
                  _offset{streamId * threadsPerStream + threadBindingOffset},
                  _cpuIdxOffset(cpuIdxOffset),
                  _cpu_ids(cpu_ids) {}
            void on_scheduler_entry(bool) override {
                pin_thread_to_vacant_core(_offset + tbb::this_task_arena::current_thread_index(),
                                          _threadBindingStep,
                                          _ncpus,
                                          _mask,
                                          _cpu_ids,
                                          _cpuIdxOffset);
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
                if (!_impl->_subStreamIdQueue.empty() && _impl->_subStreamsNum < _impl->_config._sub_streams) {
                    _sub_stream_id = _impl->_subStreamIdQueue.front();
                    _impl->_subStreamIdQueue.pop();
                    _impl->_subStreamsNum++;
                }
            }
            _numaNodeId = _impl->_config._streams
                              ? _impl->_usedNumaNodes.at((_streamId % _impl->_config._streams) /
                                                         ((_impl->_config._streams + _impl->_usedNumaNodes.size() - 1) /
                                                          _impl->_usedNumaNodes.size()))
                              : _impl->_usedNumaNodes.at(_streamId % _impl->_usedNumaNodes.size());
#if OV_THREAD == OV_THREAD_TBB || OV_THREAD == OV_THREAD_TBB_AUTO
            if (is_cpu_map_available() && _impl->_config._streams_info_table.size() > 0) {
                init_stream();
            } else {
                init_stream_legacy();
            }
#elif OV_THREAD == OV_THREAD_OMP
            omp_set_num_threads(_impl->_config._threadsPerStream);
            if (!check_open_mp_env_vars(false) && (ThreadBindingType::NONE != _impl->_config._threadBindingType)) {
                CpuSet processMask;
                int ncpus = 0;
                std::tie(processMask, ncpus) = get_process_mask();
                if (nullptr != processMask) {
                    parallel_nt(_impl->_config._threadsPerStream, [&](int threadIndex, int threadsPerStream) {
                        int thrIdx = _streamId * _impl->_config._threadsPerStream + threadIndex +
                                     _impl->_config._threadBindingOffset;
                        pin_thread_to_vacant_core(thrIdx, _impl->_config._threadBindingStep, ncpus, processMask);
                    });
                }
            }
#elif OV_THREAD == OV_THREAD_SEQ
            if (ThreadBindingType::NUMA == _impl->_config._threadBindingType) {
                pin_current_thread_to_socket(_numaNodeId);
            } else if (ThreadBindingType::CORES == _impl->_config._threadBindingType) {
                CpuSet processMask;
                int ncpus = 0;
                std::tie(processMask, ncpus) = get_process_mask();
                if (nullptr != processMask) {
                    pin_thread_to_vacant_core(_streamId + _impl->_config._threadBindingOffset,
                                              _impl->_config._threadBindingStep,
                                              ncpus,
                                              processMask);
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
            if (_impl->_config._name.find("StreamsExecutor") == std::string::npos) {
                set_cpu_used(_cpu_ids, NOT_USED);
            }
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
                                   const int max_threads_per_core) {
            _numaNodeId = std::max(0, numa_node_id);
            _socketId = get_socket_by_numa_node(_numaNodeId);
            if (stream_type == STREAM_WITHOUT_PARAM) {
                _taskArena.reset(new custom::task_arena{custom::task_arena::constraints{}
                                                            .set_max_concurrency(concurrency)
                                                            .set_max_threads_per_core(max_threads_per_core)});
            } else if (stream_type == STREAM_WITH_NUMA_ID) {
                _taskArena.reset(new custom::task_arena{custom::task_arena::constraints{}
                                                            .set_numa_id(_numaNodeId)
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
                _cpu_ids = _impl->_config._stream_processor_ids[stream_id];
                if (_cpu_ids.size() > 0) {
                    CpuSet processMask;
                    int ncpus = 0;
                    std::tie(processMask, ncpus) = get_process_mask();
                    if (nullptr != processMask) {
                        _observer.reset(new Observer{*_taskArena,
                                                     std::move(processMask),
                                                     ncpus,
                                                     0,
                                                     concurrency,
                                                     0,
                                                     0,
                                                     0,
                                                     _cpu_ids});
                        _observer->observe(true);
                    }
                }
            }
        }
        void init_stream() {
            int concurrency;
            int cpu_core_type;
            int numa_node_id;
            int max_threads_per_core;
            StreamCreateType stream_type;
            const auto org_proc_type_table = get_org_proc_type_table();
            const auto stream_id =
                _sub_stream_id >= 0 ? _impl->_config._streams + _sub_stream_id
                                    : (_streamId >= _impl->_config._streams ? _impl->_config._streams - 1 : _streamId);

            get_cur_stream_info(stream_id,
                                _impl->_config._cpu_reservation,
                                org_proc_type_table,
                                _impl->_config._streams_info_table,
                                stream_type,
                                concurrency,
                                cpu_core_type,
                                numa_node_id,
                                max_threads_per_core);
            if (concurrency <= 0) {
                return;
            }
            create_tbb_task_arena(stream_id,
                                  stream_type,
                                  concurrency,
                                  cpu_core_type,
                                  numa_node_id,
                                  max_threads_per_core);
        }

        void init_stream_legacy() {
            const auto concurrency = (0 == _impl->_config._threadsPerStream) ? custom::task_arena::automatic
                                                                             : _impl->_config._threadsPerStream;
            if (ThreadBindingType::HYBRID_AWARE == _impl->_config._threadBindingType) {
                if (Config::PreferredCoreType::ROUND_ROBIN != _impl->_config._threadPreferredCoreType) {
                    if (Config::PreferredCoreType::ANY == _impl->_config._threadPreferredCoreType) {
                        _taskArena.reset(new custom::task_arena{concurrency});
                    } else {
                        const auto selected_core_type =
                            Config::PreferredCoreType::BIG == _impl->_config._threadPreferredCoreType
                                ? custom::info::core_types().back()    // running on Big cores only
                                : custom::info::core_types().front();  // running on Little cores only
                        _taskArena.reset(new custom::task_arena{custom::task_arena::constraints{}
                                                                    .set_core_type(selected_core_type)
                                                                    .set_max_concurrency(concurrency)});
                    }
                } else {
                    // assigning the stream to the core type in the round-robin fashion
                    // wrapping around total_streams (i.e. how many streams all different core types can handle
                    // together). Binding priority: Big core, Logical big core, Small core
                    const auto total_streams = _impl->total_streams_on_core_types.back().second;
                    const auto big_core_streams = _impl->total_streams_on_core_types.front().second;
                    const auto hybrid_core = _impl->total_streams_on_core_types.size() > 1;
                    const auto phy_core_streams =
                        _impl->_config._big_core_streams == 0
                            ? 0
                            : _impl->num_big_core_phys / _impl->_config._threads_per_stream_big;
                    const auto streamId_wrapped = _streamId % total_streams;
                    const auto& selected_core_type =
                        std::find_if(
                            _impl->total_streams_on_core_types.cbegin(),
                            _impl->total_streams_on_core_types.cend(),
                            [streamId_wrapped](const decltype(_impl->total_streams_on_core_types)::value_type& p) {
                                return p.second > streamId_wrapped;
                            })
                            ->first;
                    const auto small_core = hybrid_core && selected_core_type == 0;
                    const auto logic_core = !small_core && streamId_wrapped >= phy_core_streams;
                    const auto small_core_skip = small_core && _impl->_config._threads_per_stream_small == 3 &&
                                                 _impl->_config._small_core_streams > 1;
                    const auto max_concurrency =
                        small_core ? _impl->_config._threads_per_stream_small : _impl->_config._threads_per_stream_big;
                    // Special handling of _threads_per_stream_small == 3
                    const auto small_core_id = small_core_skip ? 0 : streamId_wrapped - big_core_streams;
                    const auto stream_id =
                        hybrid_core
                            ? (small_core ? small_core_id
                                          : (logic_core ? streamId_wrapped - phy_core_streams : streamId_wrapped))
                            : streamId_wrapped;
                    const auto thread_binding_step = hybrid_core ? (small_core ? _impl->_config._threadBindingStep : 2)
                                                                 : _impl->_config._threadBindingStep;
                    // Special handling of _threads_per_stream_small == 3, need to skip 4 (Four cores share one L2
                    // cache on the small core), stream_id = 0, cpu_idx_offset cumulative plus 4
                    const auto small_core_offset =
                        small_core_skip ? _impl->_config._small_core_offset + (streamId_wrapped - big_core_streams) * 4
                                        : _impl->_config._small_core_offset;
                    const auto cpu_idx_offset =
                        hybrid_core
                            // Prevent conflicts with system scheduling, so default cpu id on big core starts from 1
                            ? (small_core ? small_core_offset : (logic_core ? 0 : 1))
                            : 0;
#    ifdef _WIN32
                    _taskArena.reset(new custom::task_arena{custom::task_arena::constraints{}
                                                                .set_core_type(selected_core_type)
                                                                .set_max_concurrency(max_concurrency)});
#    else
                    _taskArena.reset(new custom::task_arena{max_concurrency});
#    endif
                    CpuSet processMask;
                    int ncpus = 0;
                    std::tie(processMask, ncpus) = get_process_mask();
                    if (nullptr != processMask) {
                        _observer.reset(new Observer{*_taskArena,
                                                     std::move(processMask),
                                                     ncpus,
                                                     stream_id,
                                                     max_concurrency,
                                                     thread_binding_step,
                                                     _impl->_config._threadBindingOffset,
                                                     cpu_idx_offset});
                        _observer->observe(true);
                    }
                }
            } else if (ThreadBindingType::NUMA == _impl->_config._threadBindingType) {
                _taskArena.reset(new custom::task_arena{custom::task_arena::constraints{_numaNodeId, concurrency}});
            } else if ((0 != _impl->_config._threadsPerStream) ||
                       (ThreadBindingType::CORES == _impl->_config._threadBindingType)) {
                _taskArena.reset(new custom::task_arena{concurrency});
                if (ThreadBindingType::CORES == _impl->_config._threadBindingType) {
                    CpuSet processMask;
                    int ncpus = 0;
                    std::tie(processMask, ncpus) = get_process_mask();
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
        }
#endif

        Impl* _impl = nullptr;
        int _streamId = 0;
        int _numaNodeId = 0;
        int _socketId = 0;
        bool _execute = false;
        int _sub_stream_id = -1;
        std::queue<Task> _taskQueue;
#if OV_THREAD == OV_THREAD_TBB || OV_THREAD == OV_THREAD_TBB_AUTO
        std::unique_ptr<custom::task_arena> _taskArena;
        std::unique_ptr<Observer> _observer;
        std::vector<int> _cpu_ids;
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
        if (_config._streams != 0) {
            std::copy_n(std::begin(numaNodes),
                        std::min<std::size_t>(_config._streams, numaNodes.size()),
                        std::back_inserter(_usedNumaNodes));
        } else {
            _usedNumaNodes = numaNodes;
        }
#if (OV_THREAD == OV_THREAD_TBB || OV_THREAD == OV_THREAD_TBB_AUTO)
        if (!is_cpu_map_available() && ThreadBindingType::HYBRID_AWARE == config._threadBindingType) {
            const auto core_types = custom::info::core_types();
            const auto num_core_phys = get_number_of_cpu_cores();
            num_big_core_phys = get_number_of_cpu_cores(true);
            const auto num_small_core_phys = num_core_phys - num_big_core_phys;
            int sum = 0;
            // reversed order, so BIG cores are first
            for (auto iter = core_types.rbegin(); iter < core_types.rend(); iter++) {
                const auto& type = *iter;
                // calculating the #streams per core type
                const int num_streams_for_core_type =
                    type == 0 ? std::max(1,
                                         std::min(config._small_core_streams,
                                                  config._threads_per_stream_small == 0
                                                      ? 0
                                                      : num_small_core_phys / config._threads_per_stream_small))
                              : std::max(1,
                                         std::min(config._big_core_streams,
                                                  config._threads_per_stream_big == 0
                                                      ? 0
                                                      : num_big_core_phys / config._threads_per_stream_big * 2));
                sum += num_streams_for_core_type;
                // prefix sum, so the core type for a given stream id will be deduced just as a upper_bound
                // (notice that the map keeps the elements in the descending order, so the big cores are populated
                // first)
                total_streams_on_core_types.push_back({type, sum});
            }
        }
#endif
        _subTaskThread.assign(_config._sub_streams, std::make_shared<SubQueue>());
        for (auto streamId = 0; streamId < _config._streams; ++streamId) {
            _threads.emplace_back([this, streamId] {
                openvino::itt::threadName(_config._name + "_" + std::to_string(streamId));
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

        for (auto subId = 0; subId < _config._sub_streams; ++subId) {
            _subThreads.emplace_back([this, subId] {
                openvino::itt::threadName(_config._name + "_subthreads" + "_" + std::to_string(subId));
                for (bool stopped = false; !stopped;) {
                    Task task;
                    { _subTaskThread[subId]->que_pop(task, stopped); }
                    if (task) {
                        {
                            std::lock_guard<std::mutex> lock{_streamIdMutex};
                            if (_subStreamsNum < _config._sub_streams) {
                                _subStreamIdQueue.push(subId);
                            } else {
                                std::queue<int> empty;
                                std::swap(_subStreamIdQueue, empty);
                            }
                        }
                        Execute(task, *(_streams.local()));
                    }
                }
            });
        }
        if (_subThreads.size() > 0) {
            _streams.set_thread_ids_map(_subThreads);
        }
    }

    void Enqueue(Task task) {
        {
            std::lock_guard<std::mutex> lock(_mutex);
            _taskQueue.emplace(std::move(task));
        }
        _queueCondVar.notify_one();
    }

    void Enqueue_sub(Task task, int id) {
        _subTaskThread[id]->que_push(std::move(task));
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
            } catch (...) {
            }
            stream._execute = false;
        }
    }

    struct SubQueue {
        std::mutex _subMutex;
        std::condition_variable _subQueueCondVar;
        bool _isSubStopped = false;
        std::queue<Task> _subTaskQueue;

        SubQueue() {}

        void que_push(Task task) {
            {
                std::lock_guard<std::mutex> lock(_subMutex);
                _subTaskQueue.emplace(std::move(task));
            }
            _subQueueCondVar.notify_one();
        }

        void que_pop(Task& task, bool& stopped) {
            std::unique_lock<std::mutex> lock(_subMutex);
            _subQueueCondVar.wait(lock, [&] {
                return !_subTaskQueue.empty() || (stopped = _isSubStopped);
            });
            if (!_subTaskQueue.empty()) {
                task = std::move(_subTaskQueue.front());
                _subTaskQueue.pop();
            }
        }

        ~SubQueue() {}
    };

    Config _config;
    std::mutex _streamIdMutex;
    int _streamId = 0;
    std::queue<int> _streamIdQueue;
    std::queue<int> _subStreamIdQueue;
    int _subStreamsNum = 0;
    std::vector<std::thread> _threads;
    std::vector<std::thread> _subThreads;
    std::mutex _mutex;
    std::condition_variable _queueCondVar;
    std::queue<Task> _taskQueue;
    bool _isStopped = false;
    std::vector<std::shared_ptr<SubQueue>> _subTaskThread;
    std::vector<int> _usedNumaNodes;
    CustomThreadLocal _streams;
#if (OV_THREAD == OV_THREAD_TBB || OV_THREAD == OV_THREAD_TBB_AUTO)
    // stream id mapping to the core type
    // stored in the reversed order (so the big cores, with the highest core_type_id value, are populated first)
    // every entry is the core type and #streams that this AND ALL EARLIER entries can handle (prefix sum)
    // (so mapping is actually just an upper_bound: core type is deduced from the entry for which the id < #streams)
    using StreamIdToCoreTypes = std::vector<std::pair<custom::core_type_id, int>>;
    StreamIdToCoreTypes total_streams_on_core_types;
    int num_big_core_phys;
#endif
    std::shared_ptr<ExecutorManager> _exectorMgr;
};

int CPUStreamsExecutor::get_stream_id() {
    auto stream = _impl->_streams.local();
    return stream->_streamId;
}

int CPUStreamsExecutor::get_numa_node_id() {
    auto stream = _impl->_streams.local();
    return stream->_numaNodeId;
}

int CPUStreamsExecutor::get_socket_id() {
    auto stream = _impl->_streams.local();
    return stream->_socketId;
}

std::vector<int> CPUStreamsExecutor::get_cores_mt_sockets() {
    std::vector<int> cores;
    std::vector<std::vector<int>> stream_table = _impl->_config._streams_info_table;
    if (stream_table.size() > 0) {
        if (stream_table[0][NUMBER_OF_STREAMS] == 1) {  // cores in main stream
            cores.push_back(stream_table[0][THREADS_PER_STREAM]);
        } else {
            return cores;
        }
        for (size_t i = 1; i < stream_table.size(); i++) {
            if (stream_table[i][NUMBER_OF_STREAMS] < 0) {  // cores in sub stream
                cores.push_back(stream_table[i][THREADS_PER_STREAM]);
            }
        }
    }
    return cores;
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
    for (size_t i = 0; i < _impl->_subTaskThread.size(); i++) {
        {
            std::lock_guard<std::mutex> lock(_impl->_subTaskThread[i]->_subMutex);
            _impl->_subTaskThread[i]->_isSubStopped = true;
        }
        _impl->_subTaskThread[i]->_subQueueCondVar.notify_all();
    }
    for (auto& thread : _impl->_subThreads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void CPUStreamsExecutor::execute(Task task) {
    _impl->Defer(std::move(task));
}

void CPUStreamsExecutor::run(Task task) {
    if (0 == _impl->_config._streams) {
        _impl->Defer(std::move(task));
    } else {
        _impl->Enqueue(std::move(task));
    }
}

void CPUStreamsExecutor::run_id(Task task, int id) {
    _impl->Enqueue_sub(std::move(task), id);
}

}  // namespace threading
}  // namespace ov
