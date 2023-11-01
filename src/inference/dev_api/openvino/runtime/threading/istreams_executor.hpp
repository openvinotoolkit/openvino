// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @file openvino/runtime/threading/istreams_executor.hpp
 * @brief A header file for OpenVINO Streams-based Executor Interface
 */

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "openvino/runtime/common.hpp"
#include "openvino/runtime/system_conf.hpp"
#include "openvino/runtime/threading/itask_executor.hpp"

namespace ov {
namespace threading {

/**
 * @interface IStreamsExecutor
 * @ingroup ov_dev_api_threading
 * @brief Interface for Streams Task Executor. This executor groups worker threads into so-called `streams`.
 * @par CPU
 *        The executor executes all parallel tasks using threads from one stream.
 *        With proper pinning settings it should reduce cache misses for memory bound workloads.
 * @par NUMA
 *        On NUMA hosts GetNumaNodeId() method can be used to define the NUMA node of current stream
 */
class OPENVINO_RUNTIME_API IStreamsExecutor : virtual public ITaskExecutor {
public:
    /**
     * @brief Defines inference thread binding type
     */
    enum ThreadBindingType : std::uint8_t {
        NONE,   //!< Don't bind the inference threads
        CORES,  //!< Bind inference threads to the CPU cores (round-robin)
        // the following modes are implemented only for the TBB code-path:
        NUMA,  //!< Bind to the NUMA nodes (default mode for the non-hybrid CPUs on the Win/MacOS, where the 'CORES' is
               //!< not implemeneted)
        HYBRID_AWARE  //!< Let the runtime bind the inference threads depending on the cores type (default mode for the
                      //!< hybrid CPUs)
    };

    /**
     * @brief Defines IStreamsExecutor configuration
     */
    struct OPENVINO_RUNTIME_API Config {
        /**
         * @brief Sets configuration
         * @param properties map of properties
         */
        void set_property(const ov::AnyMap& properties);

        /**
         * @brief Sets configuration
         * @param key property name
         * @param value property value
         */
        void set_property(const std::string& key, const ov::Any& value);

        /**
         * @brief Return configuration value
         * @param key configuration key
         * @return configuration value wrapped into ov::Any
         */
        ov::Any get_property(const std::string& key) const;

        /**
         * @brief Create appropriate multithreaded configuration
         *        filing unconfigured values from initial configuration using hardware properties
         * @param initial Inital configuration
         * @param fp_intesive additional hint for the the (Hybrid) core-types selection logic
         *       whether the executor should be configured for floating point intensive work (as opposite to int8
         * intensive)
         * @return configured values
         */
        static Config make_default_multi_threaded(const Config& initial, const bool fp_intesive = true);
        static int get_default_num_streams(
            const bool enable_hyper_thread = true);  // no network specifics considered (only CPU's caps);
        static int get_hybrid_num_streams(std::map<std::string, std::string>& config, const int stream_mode);
        static void update_hybrid_custom_threads(Config& config);
        static Config reserve_cpu_threads(const Config& initial);

        std::string _name;          //!< Used by `ITT` to name executor threads
        int _streams = 1;           //!< Number of streams.
        int _threadsPerStream = 0;  //!< Number of threads per stream that executes `ov_parallel` calls
        ThreadBindingType _threadBindingType = ThreadBindingType::NONE;  //!< Thread binding to hardware resource type.
                                                                         //!< No binding by default
        int _threadBindingStep = 1;                                      //!< In case of @ref CORES binding offset type
                                                                         //!< thread binded to cores with defined step
        int _threadBindingOffset = 0;       //!< In case of @ref CORES binding offset type thread binded to cores
                                            //!< starting from offset
        int _threads = 0;                   //!< Number of threads distributed between streams.
                                            //!< Reserved. Should not be used.
        int _big_core_streams = 0;          //!< Number of streams in Performance-core(big core)
        int _small_core_streams = 0;        //!< Number of streams in Efficient-core(small core)
        int _threads_per_stream_big = 0;    //!< Threads per stream in big cores
        int _threads_per_stream_small = 0;  //!< Threads per stream in small cores
        int _small_core_offset = 0;         //!< Calculate small core start offset when binding cpu cores
        bool _enable_hyper_thread = true;   //!< enable hyper thread
        int _plugin_task = NOT_USED;
        enum StreamMode { DEFAULT, AGGRESSIVE, LESSAGGRESSIVE };
        enum PreferredCoreType {
            ANY,
            LITTLE,
            BIG,
            ROUND_ROBIN  // used w/multiple streams to populate the Big cores first, then the Little, then wrap around
                         // (for large #streams)
        } _threadPreferredCoreType =
            PreferredCoreType::ANY;  //!< In case of @ref HYBRID_AWARE hints the TBB to affinitize

        std::vector<std::vector<int>> _streams_info_table = {};
        std::vector<std::vector<int>> _stream_processor_ids;
        bool _cpu_reservation = false;
        bool _streams_changed = false;
        int _sub_streams = 0;

        /**
         * @brief      A constructor with arguments
         *
         * @param[in]  name                 The executor name
         * @param[in]  streams              @copybrief Config::_streams
         * @param[in]  threadsPerStream     @copybrief Config::_threadsPerStream
         * @param[in]  threadBindingType    @copybrief Config::_threadBindingType
         * @param[in]  threadBindingStep    @copybrief Config::_threadBindingStep
         * @param[in]  threadBindingOffset  @copybrief Config::_threadBindingOffset
         * @param[in]  threads              @copybrief Config::_threads
         * @param[in]  threadPreferBigCores @copybrief Config::_threadPreferBigCores
         */
        Config(std::string name = "StreamsExecutor",
               int streams = 1,
               int threadsPerStream = 0,
               ThreadBindingType threadBindingType = ThreadBindingType::NONE,
               int threadBindingStep = 1,
               int threadBindingOffset = 0,
               int threads = 0,
               PreferredCoreType threadPreferredCoreType = PreferredCoreType::ANY,
               std::vector<std::vector<int>> streamsInfoTable = {},
               bool cpuReservation = false)
            : _name{name},
              _streams{streams},
              _threadsPerStream{threadsPerStream},
              _threadBindingType{threadBindingType},
              _threadBindingStep{threadBindingStep},
              _threadBindingOffset{threadBindingOffset},
              _threads{threads},
              _threadPreferredCoreType(threadPreferredCoreType),
              _streams_info_table{streamsInfoTable},
              _cpu_reservation{cpuReservation} {}
    };

    /**
     * @brief A virtual destructor
     */
    ~IStreamsExecutor() override;

    /**
     * @brief Return the index of current stream
     * @return An index of current stream. Or throw exceptions if called not from stream thread
     */
    virtual int get_stream_id() = 0;

    /**
     * @brief Return the id of current NUMA Node
     *        Return 0 when current stream cross some NUMA Nodes
     * @return `ID` of current NUMA Node, or throws exceptions if called not from stream thread
     */
    virtual int get_numa_node_id() = 0;

    /**
     * @brief Return the id of current socket
     *        Return 0 when current stream cross some sockets
     * @return `ID` of current socket, or throws exceptions if called not from stream thread
     */
    virtual int get_socket_id() = 0;

    /**
     * @brief Return numbers of cores in different sockets
     * @return Numbers of cores per socket, or return empty vector
     */
    virtual std::vector<int> get_cores_mt_sockets() = 0;

    /**
     * @brief Execute the task in the current thread using streams executor configuration and constraints
     * @param task A task to start
     */
    virtual void execute(Task task) = 0;

    /**
     * @brief Execute ov::Task inside sub stream of task executor context
     * @param task A task to start
     * @param id Sub stream id
     */
    virtual void run_id(Task task, int id) = 0;

    /**
     * @brief Execute all of the tasks and waits for its completion.
     *        Default run_and_wait_id() method implementation uses run_id() pure virtual method
     *        and higher level synchronization primitives from STL.
     *        The task is wrapped into std::packaged_task which returns std::future.
     *        std::packaged_task will call the task and signal to std::future that the task is finished
     *        or the exception is thrown from task
     *        Than std::future is used to wait for task execution completion and
     *        task exception extraction
     * @note run_and_wait_id() does not copy or capture tasks!
     * @param tasks A vector of tasks to execute
     * @param id Sub stream id
     */
    void run_and_wait_id(const std::vector<Task>& tasks, int id);
};

}  // namespace threading
}  // namespace ov
