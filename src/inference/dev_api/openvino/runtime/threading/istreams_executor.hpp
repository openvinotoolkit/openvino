// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @file ie_istreams_executor.hpp
 * @brief A header file for Inference Engine Streams-based Executor Interface
 */

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "openvino/runtime/common.hpp"
#include "openvino/runtime/property_supervisor.hpp"
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
    class OPENVINO_RUNTIME_API Config {
    public:
        enum PreferredCoreType {
            ANY,
            LITTLE,
            BIG,
            ROUND_ROBIN  // used w/multiple streams to populate the Big cores first, then the Little, then wrap around
                         // (for large #streams)
        };
        /********************** Config properties **********************/
        static constexpr Property<std::string, PropertyMutability::RW> name{"ISTREAM_EXECUTOR_CONFIG_NAME"};
        static constexpr Property<int32_t, PropertyMutability::RW> streams{"ISTREAM_EXECUTOR_CONFIG_STREAMS"};
        static constexpr Property<int32_t, PropertyMutability::RW> threads_per_stream{
            "ISTREAM_EXECUTOR_CONFIG_THREADS_PER_STREAM"};
        static constexpr Property<ThreadBindingType, PropertyMutability::RW> thread_binding_type{
            "ISTREAM_EXECUTOR_CONFIG_THREAD_BINDING_TYPE"};
        static constexpr Property<int32_t, PropertyMutability::RW> thread_binding_step{
            "ISTREAM_EXECUTOR_CONFIG_THREAD_BINDING_STEP"};
        static constexpr Property<int32_t, PropertyMutability::RW> thread_binding_offset{
            "ISTREAM_EXECUTOR_CONFIG_THREAD_BINDING_OFFSET"};
        static constexpr Property<int32_t, PropertyMutability::RW> threads{"ISTREAM_EXECUTOR_CONFIG_THREADS"};
        static constexpr Property<int32_t, PropertyMutability::RW> big_core_streams{
            "ISTREAM_EXECUTOR_CONFIG_BIG_CORE_STREAMS"};
        static constexpr Property<int32_t, PropertyMutability::RW> small_core_streams{
            "ISTREAM_EXECUTOR_CONFIG_SMALL_CORE_STREAMS"};
        static constexpr Property<int32_t, PropertyMutability::RW> threads_per_stream_big{
            "ISTREAM_EXECUTOR_CONFIG_THREADS_PER_STREAM_BIG"};
        static constexpr Property<int32_t, PropertyMutability::RW> threads_per_stream_small{
            "ISTREAM_EXECUTOR_CONFIG_THREADS_PER_STREAM_SMALL"};
        static constexpr Property<int32_t, PropertyMutability::RW> small_core_offset{
            "ISTREAM_EXECUTOR_CONFIG_SMALL_CORE_OFFSET"};
        static constexpr Property<bool, PropertyMutability::RW> enable_hyper_thread{
            "ISTREAM_EXECUTOR_CONFIG_ENABLE_HYPER_THREAD"};
        static constexpr Property<PreferredCoreType, PropertyMutability::RW> thread_preferred_core_type{
            "ISTREAM_EXECUTOR_CONFIG_THREAD_PREFERRED_CORE_TYPE"};
        /********************** Config properties **********************/
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

        /**
         * @brief A constructor with config
         *
         * @param config
         */
        Config(const std::string& name, const ov::AnyMap& config = {});

    private:
        enum StreamMode { DEFAULT, AGGRESSIVE, LESSAGGRESSIVE };
        ov::PropertySupervisor m_properties;
        std::string m_name = "StreamsExecutor";  //!< Used by `ITT` to name executor threads
        int32_t m_streams = 1;                   //!< Number of streams.
        int32_t m_threads_per_stream = 0;        //!< Number of threads per stream that executes `ie_parallel` calls
        ThreadBindingType m_thread_binding_type = ThreadBindingType::NONE;  //!< Thread binding to hardware resource
                                                                            //!< type. No binding by default
        int32_t m_thread_binding_step = 1;       //!< In case of @ref CORES binding offset type
                                                 //!< thread binded to cores with defined step
        int32_t m_thread_binding_offset = 0;     //!< In case of @ref CORES binding offset type thread binded to cores
                                                 //!< starting from offset
        int32_t m_threads = 0;                   //!< Number of threads distributed between streams.
                                                 //!< Reserved. Should not be used.
        int32_t m_big_core_streams = 0;          //!< Number of streams in Performance-core(big core)
        int32_t m_small_core_streams = 0;        //!< Number of streams in Efficient-core(small core)
        int32_t m_threads_per_stream_big = 0;    //!< Threads per stream in big cores
        int32_t m_threads_per_stream_small = 0;  //!< Threads per stream in small cores
        int32_t m_small_core_offset = 0;         //!< Calculate small core start offset when binding cpu cores
        bool m_enable_hyper_thread = true;       //!< enable hyper thread
        PreferredCoreType m_thread_preferred_core_type =
            PreferredCoreType::ANY;  //!< In case of @ref HYBRID_AWARE hints the TBB to affinitize
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
     * @return `ID` of current NUMA Node, or throws exceptions if called not from stream thread
     */
    virtual int get_numa_node_id() = 0;

    /**
     * @brief Execute the task in the current thread using streams executor configuration and constraints
     * @param task A task to start
     */
    virtual void execute(Task task) = 0;
};

}  // namespace threading
}  // namespace ov
