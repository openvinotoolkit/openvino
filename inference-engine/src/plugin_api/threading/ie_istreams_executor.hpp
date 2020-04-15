// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @file ie_istreams_executor.hpp
 * @brief A header file for Inference Engine Streams-based Executor Interface
 */

#pragma once

#include <memory>
#include "threading/ie_itask_executor.hpp"
#include "ie_api.h"
#include "ie_parameter.hpp"
#include <vector>
#include <string>

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
     * @brief Defines thread binding type
     */
    enum ThreadBindingType : std::uint8_t {
        NONE,    //!< Don't bind threads
        CORES,   //!< Bind threads to cores
        NUMA     //!< Bind threads to NUMA nodes
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

        std::string        _name;  //!< Used by `ITT` to name executor threads
        int                _streams                 = 1;  //!< Number of streams.
        int                _threadsPerStream        = 0;  //!< Number of threads per stream that executes `ie_parallel` calls
        ThreadBindingType  _threadBindingType       = ThreadBindingType::NONE;  //!< Thread binding to hardware resource type. No binding by default
        int                _threadBindingStep       = 1;  //!< In case of @ref CORES binding offset type thread binded to cores with defined step
        int                _threadBindingOffset     = 0;  //!< In case of @ref CORES binding offset type thread binded to cores starting from offset
        int                _threads                 = 0;  //!< Number of threads distributed between streams. Reserved. Should not be used.

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
         */
        Config(
            std::string        name                    = "StreamsExecutor",
            int                streams                 = 1,
            int                threadsPerStream        = 0,
            ThreadBindingType  threadBindingType       = ThreadBindingType::NONE,
            int                threadBindingStep       = 1,
            int                threadBindingOffset     = 0,
            int                threads                 = 0) :
        _name{name},
        _streams{streams},
        _threadsPerStream{threadsPerStream},
        _threadBindingType{threadBindingType},
        _threadBindingStep{threadBindingStep},
        _threadBindingOffset{threadBindingOffset},
        _threads{threads} {
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
};



}  // namespace InferenceEngine
