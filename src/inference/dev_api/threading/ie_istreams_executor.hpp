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

#include "ie_parameter.hpp"
#include "openvino/runtime/threading/istreams_executor.hpp"
#include "threading/ie_itask_executor.hpp"

namespace InferenceEngine {

/**
 * @interface IStreamsExecutor
 * @ingroup ie_dev_api_threading
 * @brief Interface for Streams Task Executor. This executor groups worker threads into so-called `streams`.
 * @par CPU
 *        The executor executes all parallel tasks using threads from one stream.
 *        With proper pinning settings it should reduce cache misses for memory bound workloads.
 * @par NUMA
 *        On NUMA hosts GetNumaNodeId() method can be used to define the NUMA node of current stream
 */
class INFERENCE_ENGINE_API_CLASS(IStreamsExecutor) : public ITaskExecutor, public ov::threading::IStreamsExecutor {
public:
    /**
     * A shared pointer to IStreamsExecutor interface
     */
    using Ptr = std::shared_ptr<IStreamsExecutor>;

    /**
     * @brief Defines IStreamsExecutor configuration
     */
    struct INFERENCE_ENGINE_API_CLASS(Config) : public ov::threading::IStreamsExecutor::Config {
        /**
         * @brief Supported Configuration keys
         * @return vector of supported configuration keys
         */
        std::vector<std::string> SupportedKeys() const;

        /**
         * @brief Parses configuration key/value pair
         * @param key configuration key
         * @param value configuration values
         */
        void SetConfig(const std::string& key, const std::string& value);

        /**
         * @brief Return configuration value
         * @param key configuration key
         * @return configuration value wrapped into Parameter
         */
        Parameter GetConfig(const std::string& key) const;

        /**
         * @brief Create appropriate multithreaded configuration
         *        filing unconfigured values from initial configuration using hardware properties
         * @param initial Inital configuration
         * @param fp_intesive additional hint for the the (Hybrid) core-types selection logic
         *       whether the executor should be configured for floating point intensive work (as opposite to int8
         * intensive)
         * @return configured values
         */
        static Config MakeDefaultMultiThreaded(const Config& initial, const bool fp_intesive = true);
        static int GetDefaultNumStreams(
            const bool enable_hyper_thread = true);  // no network specifics considered (only CPU's caps);
        static int GetHybridNumStreams(std::map<std::string, std::string>& config, const int stream_mode);
        static void UpdateHybridCustomThreads(Config& config);
        static Config ReserveCpuThreads(const Config& initial);

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
            : ov::threading::IStreamsExecutor::Config(name,
                                                      streams,
                                                      threadsPerStream,
                                                      threadBindingType,
                                                      threadBindingStep,
                                                      threadBindingOffset,
                                                      threads,
                                                      threadPreferredCoreType,
                                                      streamsInfoTable,
                                                      cpuReservation) {}

        Config(const ov::threading::IStreamsExecutor::Config& config)
            : ov::threading::IStreamsExecutor::Config(config) {}
    };

    /**
     * @brief A virtual destructor
     */
    ~IStreamsExecutor() override;

    /**
     * @brief Return the index of current stream
     * @return An index of current stream. Or throw exceptions if called not from stream thread
     */
    virtual int GetStreamId() = 0;

    /**
     * @brief Return the id of current NUMA Node
     * @return `ID` of current NUMA Node, or throws exceptions if called not from stream thread
     */
    virtual int GetNumaNodeId() = 0;

    /**
     * @brief Return the id of current socket
     * @return `ID` of current socket, or throws exceptions if called not from stream thread
     */
    virtual int GetSocketId() = 0;

    /**
     * @brief Execute the task in the current thread using streams executor configuration and constraints
     * @param task A task to start
     */
    virtual void Execute(Task task) = 0;

    int get_stream_id() override {
        return GetStreamId();
    }

    int get_numa_node_id() override {
        return GetNumaNodeId();
    }

    int get_socket_id() override {
        return GetSocketId();
    }

    void execute(Task task) override {
        Execute(task);
    }
};

}  // namespace InferenceEngine
