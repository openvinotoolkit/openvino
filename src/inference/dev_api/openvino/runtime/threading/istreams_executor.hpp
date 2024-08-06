// Copyright (C) 2018-2024 Intel Corporation
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
#include "openvino/runtime/properties.hpp"
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
     * A shared pointer to IStreamsExecutor interface
     */
    using Ptr = std::shared_ptr<IStreamsExecutor>;

    enum MsgType{
        TP,
        START_INFER,
        CALL_BACK
    };

    struct MessageInfo{
        MsgType msg_type;
        std::vector<int> rank;
        void* buf;
        Task task;
    };

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
    public:
        enum PreferredCoreType {
            ANY,
            LITTLE,
            BIG,
            ROUND_ROBIN  // used w/multiple streams to populate the Big cores first, then the Little, then wrap around
                         // (for large #streams)
        };

        /**
         * @enum       StreamsMode
         * @brief      This enum contains definition of each sub streams mode, indicating the main stream situation.
         */
        enum class StreamsMode {
            SUB_STREAMS_NULL,        //!< Do not create sub streams
            SUB_STREAMS_FOR_SOCKET,  //!< Create sub streams for multiple sockets in main stream
            LATENCY,                 //!< latency mode
            THROUGHPUT,              //!< throughput mode
        };

    private:
        std::string _name;             //!< Used by `ITT` to name executor threads
        int _streams = 1;              //!< Number of streams.
        int _threads_per_stream = 0;   //!< Number of threads per stream that executes `ov_parallel` calls
        int _threadBindingStep = 1;    //!< In case of @ref CORES binding offset type
                                       //!< thread binded to cores with defined step
        int _threadBindingOffset = 0;  //!< In case of @ref CORES binding offset type thread binded to cores
                                       //!< starting from offset
        int _threads = 0;              //!< Number of threads distributed between streams.
                                       //!< Reserved. Should not be used.
        ov::hint::SchedulingCoreType _thread_preferred_core_type =
            ov::hint::SchedulingCoreType::ANY_CORE;  //!< PCORE_ONLY and ECORE_ONLY are valid in hybrid core machine,
                                                     //!< ANY_CORE is valid in all machines. Core type priority:
                                                     //!< physical PCore, ECore, logical PCore
        bool _cpu_reservation = false;  //!< Whether to reserve current cores which will not be used by other plugin.
                                        //!< If it is true, cpu_pinning defaults to true.
        bool _cpu_pinning = false;      //!< Whether to bind threads to cores.
        std::vector<std::vector<int>> _streams_info_table = {};
        std::vector<std::vector<int>> _stream_processor_ids;
        int _sub_streams = 0;
        std::vector<int> _rank = {};

        /**
         * @brief Get and reserve cpu ids based on configuration and hardware information,
         *        streams_info_table must be present in the configuration
         */
        void reserve_cpu_threads();

         /**
         * @brief Modify _streams_info_table and related configuration according to configuration
         */
        void update_executor_config();

        /**
         * @brief Set _streams_info_table and _cpu_reservation in cpu streams executor config when nstreams = 0,
         *        that is, only create one thread with TBB
         */
        void set_config_zero_stream();

    public:
        /**
         * @brief      A constructor with arguments
         *
         * @param[in]  name                         The executor name
         * @param[in]  streams                      @copybrief Config::_streams
         * @param[in]  threads_per_stream           @copybrief Config::_threads_per_stream
         * @param[in]  thread_preferred_core_type   @copybrief Config::_thread_preferred_core_type
         * @param[in]  cpu_reservation              @copybrief Config::_cpu_reservation
         * @param[in]  cpu_pinning                  @copybrief Config::_cpu_pinning
         * @param[in]  streams_info_table           @copybrief Config::_streams_info_table
         * @param[in]  rank                         @copybrief Config::_rank
         */
        Config(std::string name = "StreamsExecutor",
               int streams = 1,
               int threads_per_stream = 0,
               ov::hint::SchedulingCoreType thread_preferred_core_type = ov::hint::SchedulingCoreType::ANY_CORE,
               bool cpu_reservation = false,
               bool cpu_pinning = false,
               std::vector<std::vector<int>> streams_info_table = {},
               std::vector<int> rank = {})
            : _name{std::move(name)},
              _streams{streams},
              _threads_per_stream{threads_per_stream},
              _thread_preferred_core_type(thread_preferred_core_type),
              _cpu_reservation{cpu_reservation},
              _cpu_pinning{cpu_pinning},
              _streams_info_table{std::move(streams_info_table)},
              _rank{rank} {
            update_executor_config();
        }

        // These APIs which includes set_property and get_property can not be removed until they will never be called by
        // other plugins such as NV plugin.
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

        std::string get_name() const {
            return _name;
        }
        int get_streams() const {
            return _streams;
        }
        int get_threads() const {
            return _threads;
        }
        int get_threads_per_stream() const {
            return _threads_per_stream;
        }
        bool get_cpu_reservation() const {
            return _cpu_reservation;
        }
        bool get_cpu_pinning() const {
            return _cpu_pinning;
        }
        std::vector<std::vector<int>> get_streams_info_table() const {
            return _streams_info_table;
        }
        std::vector<std::vector<int>> get_stream_processor_ids() const {
            return _stream_processor_ids;
        }
        int get_thread_binding_step() const {
            return _threadBindingStep;
        }
        int get_thread_binding_offset() const {
            return _threadBindingOffset;
        }
        int get_sub_streams() const {
            return _sub_streams;
        }
        std::vector<int> get_rank() const {
            return _rank;
        }
        StreamsMode get_sub_stream_mode() const {
            const auto proc_type_table = get_proc_type_table();
            int sockets = proc_type_table.size() > 1 ? static_cast<int>(proc_type_table.size()) - 1 : 1;
            return _sub_streams > 0 ? StreamsMode::SUB_STREAMS_FOR_SOCKET
                                    : (_streams <= sockets ? StreamsMode::LATENCY : StreamsMode::THROUGHPUT);
        }
        bool operator==(const Config& config) {
            if (_name == config._name && _streams == config._streams &&
                _threads_per_stream == config._threads_per_stream &&
                _thread_preferred_core_type == config._thread_preferred_core_type) {
                return true;
            } else {
                return false;
            }
        }

        /**
         * @brief Create appropriate multithreaded configuration
         *        filing unconfigured values from initial configuration using hardware properties
         * @param initial Inital configuration
         * @return configured values
         */
        static Config make_default_multi_threaded(const Config& initial);

        static int get_default_num_streams();  // no network specifics considered (only CPU's caps);
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
     * @brief Return the rank of current stream
     *        Return {} when current stream has no rank
     * @return Rank array, or throws exceptions if called not from stream thread
     */
    virtual std::vector<int> get_rank() = 0;

    /**
     * @brief Execute the task in the current thread using streams executor configuration and constraints
     * @param task A task to start
     */
    virtual void execute(Task task) = 0;
};

}  // namespace threading
}  // namespace ov
