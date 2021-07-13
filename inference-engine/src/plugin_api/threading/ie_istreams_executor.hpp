// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @file ie_istreams_executor.hpp
 * @brief A header file for Inference Engine Streams-based Executor Interface
 */

#pragma once

#include <memory>
#include <vector>
#include <string>

#include "ie_parameter.hpp"
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
class INFERENCE_ENGINE_API_CLASS(IStreamsExecutor) : public ITaskExecutor {
public:
    /**
     * A shared pointer to IStreamsExecutor interface
     */
    using Ptr = std::shared_ptr<IStreamsExecutor>;

    /**
     * @brief Defines inference thread binding type
     */
    enum ThreadBindingType : std::uint8_t {
        NONE,    //!< Don't bind the inference threads
        CORES,   //!< Bind inference threads to the CPU cores (round-robin)
        // the following modes are implemented only for the TBB code-path:
        NUMA,    //!< Bind to the NUMA nodes (default mode for the non-hybrid CPUs on the Win/MacOS, where the 'CORES' is not implemeneted)
        HYBRID_AWARE  //!< Let the runtime bind the inference threads depending on the cores type (default mode for the hybrid CPUs)
    };

    /**
     * @brief Defines IStreamsExecutor configuration
     */
    struct INFERENCE_ENGINE_API_CLASS(Config) {
        /**
        * @brief Supported Configuration keys
        * @return vector of supported configuration keys
        */
        std::vector<std::string> SupportedKeys();

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
        Parameter GetConfig(const std::string& key);

        /**
        * @brief Create appropriate multithreaded configuration
        *        filing unconfigured values from initial configuration using hardware properties
        * @param initial Inital configuration
        * @param fp_intesive additional hint for the the (Hybrid) core-types selection logic
         *       whether the executor should be configured for floating point intensive work (as opposite to int8 intensive)
        * @return configured values
        */
        static Config MakeDefaultMultiThreaded(const Config& initial, const bool fp_intesive = true);
        static int GetDefaultNumStreams(); // no network specifics considered (only CPU's caps);

        std::string        _name;  //!< Used by `ITT` to name executor threads
        int                _streams                 = 1;  //!< Number of streams.
        int                _threadsPerStream        = 0;  //!< Number of threads per stream that executes `ie_parallel` calls
        ThreadBindingType  _threadBindingType       = ThreadBindingType::NONE;  //!< Thread binding to hardware resource type. No binding by default
        int                _threadBindingStep       = 1;  //!< In case of @ref CORES binding offset type thread binded to cores with defined step
        int                _threadBindingOffset     = 0;  //!< In case of @ref CORES binding offset type thread binded to cores starting from offset
        int                _threads                 = 0;  //!< Number of threads distributed between streams. Reserved. Should not be used.
        enum PreferredCoreType {
            ANY,
            LITTLE,
            BIG,
            ROUND_ROBIN // used w/multiple streams to populate the Big cores first, then the Little, then wrap around (for large #streams)
        }                  _threadPreferredCoreType = PreferredCoreType::ANY; //!< In case of @ref HYBRID_AWARE hints the TBB to affinitize

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
        Config(
            std::string        name                    = "StreamsExecutor",
            int                streams                 = 1,
            int                threadsPerStream        = 0,
            ThreadBindingType  threadBindingType       = ThreadBindingType::NONE,
            int                threadBindingStep       = 1,
            int                threadBindingOffset     = 0,
            int                threads                 = 0,
            PreferredCoreType  threadPreferredCoreType = PreferredCoreType::ANY) :
        _name{name},
        _streams{streams},
        _threadsPerStream{threadsPerStream},
        _threadBindingType{threadBindingType},
        _threadBindingStep{threadBindingStep},
        _threadBindingOffset{threadBindingOffset},
        _threads{threads}, _threadPreferredCoreType(threadPreferredCoreType){
        }
    };

    /**
     * @brief A virtual destructor
     */
    ~IStreamsExecutor() override;

    /**
    * @brief Return the index of current stream
    * @return An index of current stream. Or throw exceptions if called not from stream thread
    */
    virtual int  GetStreamId() = 0;

    /**
    * @brief Return the id of current NUMA Node
    * @return `ID` of current NUMA Node, or throws exceptions if called not from stream thread
    */
    virtual int  GetNumaNodeId() = 0;

    /**
    * @brief Execute the task in the current thread using streams executor configuration and constraints
    * @param task A task to start
    */
    virtual void Execute(Task task) = 0;
};



}  // namespace InferenceEngine
