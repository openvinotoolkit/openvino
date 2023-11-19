// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/threading/tbb_streams_executor.hpp"

#include <atomic>
#include <list>
#include <memory>
#include <queue>
#include <thread>
#include <tuple>
#include <utility>

#include "details/ie_exception.hpp"
#include "ie_parallel.hpp"
#include "parallel_custom_arena.hpp"
#include "ie_system_conf.h"
#include "thread_affinity.hpp"

#if ((IE_THREAD == IE_THREAD_TBB) || (IE_THREAD == IE_THREAD_TBB_AUTO))
#    include <tbb/concurrent_queue.h>
#    include <tbb/enumerable_thread_specific.h>
#    ifndef TBB_PREVIEW_GLOBAL_CONTROL
#        define TBB_PREVIEW_GLOBAL_CONTROL 1
#    endif
#    include <tbb/global_control.h>
#    include <tbb/task_group.h>
#    include <tbb/task_scheduler_observer.h>

namespace ov {
namespace threading {
struct TBBStreamsExecutor::Impl {
    struct Stream;
    using TaskQueue = tbb::concurrent_queue<Task>;
    using StreamQueue = tbb::concurrent_bounded_queue<Stream*>;
    using LocalStreams = tbb::enumerable_thread_specific<Stream*>;
    struct Shared : public std::enable_shared_from_this<Shared> {
        using Ptr = std::shared_ptr<Shared>;
        TaskQueue _taskQueue;
        StreamQueue _streamQueue;
    };
    struct Stream {
        struct Observer : tbb::task_scheduler_observer {
            Stream* _thisStream = nullptr;
            LocalStreams* _localStream = nullptr;
            CpuSet _mask;
            int _ncpus = 0;
            int _threadBindingStep = 0;
            int _offset = 0;

            Observer(custom::task_arena& arena,
                     Stream* thisStream,
                     LocalStreams* localStream,
                     const bool pinToCores,
                     const int streamId,
                     const int threadsPerStream,
                     const int threadBindingStep,
                     const int threadBindingOffset)
                : tbb::task_scheduler_observer{static_cast<tbb::task_arena&>(arena)},
                  _thisStream{thisStream},
                  _localStream{localStream},
                  _threadBindingStep{threadBindingStep},
                  _offset{streamId * threadsPerStream + threadBindingOffset} {
                if (pinToCores) {
                    std::tie(_mask, _ncpus) = get_process_mask();
                }
            }
            void on_scheduler_entry(bool) override {
                _localStream->local() = _thisStream;
                if (nullptr != _mask) {
                    pin_thread_to_vacant_core(_offset + tbb::this_task_arena::current_thread_index(),
                                              _threadBindingStep,
                                              _ncpus,
                                              _mask);
                }
            }
            void on_scheduler_exit(bool) override {
                _localStream->local() = nullptr;
                if (nullptr != _mask) {
                    pin_current_thread_by_mask(_ncpus, _mask);
                }
            }
            ~Observer() override = default;
        };

        explicit Stream(Impl* impl, const bool externStream = false) : _impl{impl} {
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
                              ? _impl->_usedNumaNodes.at((_streamId % _impl->_config._streams) /
                                                         ((_impl->_config._streams + _impl->_usedNumaNodes.size() - 1) /
                                                          _impl->_usedNumaNodes.size()))
                              : _impl->_usedNumaNodes.at(_streamId % _impl->_usedNumaNodes.size());
            auto concurrency =
                (0 == _impl->_config._threadsPerStream) ? tbb::task_arena::automatic : _impl->_config._threadsPerStream;
            auto masterThreads = externStream ? 1u : 0u;
            if (ThreadBindingType::HYBRID_AWARE == _impl->_config._threadBindingType) {
                if (Config::PreferredCoreType::ROUND_ROBIN != _impl->_config._threadPreferredCoreType) {
                    if (Config::PreferredCoreType::ANY == _impl->_config._threadPreferredCoreType) {
                        _arena.initialize(concurrency);
                    } else {
                        const auto selected_core_type =
                            Config::PreferredCoreType::BIG == _impl->_config._threadPreferredCoreType
                                ? custom::info::core_types().back()    // running on Big cores only
                                : custom::info::core_types().front();  // running on Little cores only
                        _arena.initialize(custom::task_arena::constraints{}
                                              .set_core_type(selected_core_type)
                                              .set_max_concurrency(concurrency));
                    }
                } else {
                    // assigning the stream to the core type in the round-robin fashion
                    // wrapping around total_streams (i.e. how many streams all different core types can handle
                    // together)
                    const auto total_streams = _impl->_totalSreamsOnCoreTypes.back().second;
                    const auto streamId_wrapped = _streamId % total_streams;
                    const auto& selected_core_type =
                        std::find_if(_impl->_totalSreamsOnCoreTypes.cbegin(),
                                     _impl->_totalSreamsOnCoreTypes.cend(),
                                     [streamId_wrapped](const decltype(_impl->_totalSreamsOnCoreTypes)::value_type& p) {
                                         return p.second > streamId_wrapped;
                                     })
                            ->first;
                    _arena.initialize(custom::task_arena::constraints{}
                                          .set_core_type(selected_core_type)
                                          .set_max_concurrency(concurrency));
                }
            } else if (ThreadBindingType::NUMA == _impl->_config._threadBindingType) {
                _arena.initialize(custom::task_arena::constraints{_numaNodeId, concurrency});
            } else {
                _arena.initialize(concurrency, masterThreads);
            }
            _observer.reset(new Observer{_arena,
                                         this,
                                         &(_impl->_localStream),
                                         (ThreadBindingType::CORES == _impl->_config._threadBindingType),
                                         _streamId,
                                         _impl->_config._threadsPerStream,
                                         _impl->_config._threadBindingStep,
                                         _impl->_config._threadBindingOffset});
            _observer->observe(true);
        }

        ~Stream() {
            static_cast<tbb::task_arena&>(_arena).terminate();
            _observer->observe(false);
            {
                std::lock_guard<std::mutex> lock{_impl->_streamIdMutex};
                _impl->_streamIdQueue.push(_streamId);
            }
        }

        Impl* _impl = nullptr;
        int _streamId = 0;
        int _numaNodeId = 0;
        custom::task_arena _arena;
        std::unique_ptr<Observer> _observer;
    };

    using Streams = std::list<Stream>;
    using ExternStreams = tbb::enumerable_thread_specific<Stream>;

    explicit Impl(const Config& config)
        : _config{config},
          _shared{std::make_shared<Shared>()},
          _localStream{nullptr},
          _externStreams{this, true} {
        if (_config._streams * _config._threadsPerStream >= static_cast<int>(std::thread::hardware_concurrency())) {
            _maxTbbThreads.reset(
                new tbb::global_control{tbb::global_control::max_allowed_parallelism,
                                        static_cast<std::size_t>(_config._streams * _config._threadsPerStream + 1)});
        }
        auto numaNodes = get_available_numa_nodes();
        if (_config._streams != 0) {
            std::copy_n(std::begin(numaNodes),
                        std::min(static_cast<std::size_t>(_config._streams), numaNodes.size()),
                        std::back_inserter(_usedNumaNodes));
        } else {
            _usedNumaNodes = numaNodes;
        }
        if (ThreadBindingType::HYBRID_AWARE == config._threadBindingType) {
            const auto core_types = custom::info::core_types();
            const int threadsPerStream =
                (0 == config._threadsPerStream) ? std::thread::hardware_concurrency() : config._threadsPerStream;
            int sum = 0;
            // reversed order, so BIG cores are first
            for (auto iter = core_types.rbegin(); iter < core_types.rend(); iter++) {
                const auto& type = *iter;
                // calculating the #streams per core type
                const int num_streams_for_core_type =
                    std::max(1,
                             custom::info::default_concurrency(custom::task_arena::constraints{}.set_core_type(type)) /
                                 threadsPerStream);
                sum += num_streams_for_core_type;
                // prefix sum, so the core type for a given stream id will be deduced just as a upper_bound
                // (notice that the map keeps the elements in the descending order, so the big cores are populated
                // first)
                _totalSreamsOnCoreTypes.emplace_back(type, sum);
            }
        }
        _shared->_streamQueue.set_capacity(_config._streams);
        for (int streamId = 0; streamId < _config._streams; ++streamId) {
            _streams.emplace_back(this);
            _shared->_streamQueue.push(&(_streams.back()));
        }
    }

    ~Impl() {
        for (int streamId = 0; streamId < _config._streams; ++streamId) {
            Stream* stream = nullptr;
            _shared->_streamQueue.pop(stream);
            (void)stream;
        }
    }

    static void Schedule(Shared::Ptr& shared, Task task) {
        Stream* stream = nullptr;
        if (shared->_streamQueue.try_pop(stream)) {
            struct TryPop {
                void operator()() const {
                    try {
                        do {
                            Task task = std::move(_task);
                            task();
                        } while (_shared->_taskQueue.try_pop(_task));
                    } catch (...) {
                    }
                    if (_shared->_streamQueue.try_push(_stream)) {
                        if (_shared->_taskQueue.try_pop(_task)) {
                            Schedule(_shared, std::move(_task));
                        }
                    }
                }
                Stream* _stream;
                mutable Shared::Ptr _shared;
                mutable Task _task;
            };
            stream->_arena.enqueue(TryPop{stream, shared->shared_from_this(), std::move(task)});
        } else {
            shared->_taskQueue.push(std::move(task));
        }
    }

    Config _config;
    std::unique_ptr<tbb::global_control> _maxTbbThreads;
    std::mutex _streamIdMutex;
    int _streamId = 0;
    std::queue<int> _streamIdQueue;
    std::vector<int> _usedNumaNodes;
    Shared::Ptr _shared;
    LocalStreams _localStream;
    ExternStreams _externStreams;
    Streams _streams;
    using StreamIdToCoreTypes = std::vector<std::pair<custom::core_type_id, int>>;
    StreamIdToCoreTypes _totalSreamsOnCoreTypes;
};

TBBStreamsExecutor::TBBStreamsExecutor(const Config& config) : _impl{new TBBStreamsExecutor::Impl{config}} {}

TBBStreamsExecutor::~TBBStreamsExecutor() {
    _impl.reset();
}

int TBBStreamsExecutor::get_stream_id() {
    auto stream = _impl->_localStream.local();
    if (nullptr == stream) {
        stream = &(_impl->_externStreams.local());
    }
    return stream->_streamId;
}

int TBBStreamsExecutor::get_numa_node_id() {
    auto stream = _impl->_localStream.local();
    if (nullptr == stream) {
        stream = &(_impl->_externStreams.local());
    }
    return stream->_numaNodeId;
}

void TBBStreamsExecutor::run(Task task) {
    if (_impl->_config._streams == 0) {
        execute(std::move(task));
    } else {
        Impl::Schedule(_impl->_shared, std::move(task));
    }
}

void TBBStreamsExecutor::execute(Task task) {
    auto stream = _impl->_localStream.local();
    if (nullptr == stream) {
        _impl->_externStreams.local()._arena.execute(std::move(task));
    } else {
        stream->_arena.execute(std::move(task));
    }
}

}  // namespace threading
}  // namespace ov
#endif  //  ((IE_THREAD == IE_THREAD_TBB) || (IE_THREAD == IE_THREAD_TBB_AUTO))
