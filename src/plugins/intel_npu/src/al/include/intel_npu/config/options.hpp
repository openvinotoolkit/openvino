// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <thread>

#include "intel_npu/config/config.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/properties.hpp"

namespace intel_npu {

struct PERFORMANCE_HINT final : OptionBase<PERFORMANCE_HINT, ov::hint::PerformanceMode> {
    static std::string_view key() {
        return ov::hint::performance_mode.name();
    }

    static constexpr std::string_view getTypeName() {
        return "ov::hint::PerformanceMode";
    }

    static ov::hint::PerformanceMode defaultValue() {
        return ov::hint::PerformanceMode::LATENCY;
    }

    static bool isPublic() {
        return true;
    }

    static ov::PropertyMutability mutability() {
        return ov::PropertyMutability::RW;
    }

    static OptionMode mode() {
        return OptionMode::Both;
    }

    static ov::hint::PerformanceMode parse(std::string_view val) {
        if (val.empty()) {
            return ov::hint::PerformanceMode::LATENCY;
        } else if (val == "LATENCY") {
            return ov::hint::PerformanceMode::LATENCY;
        } else if (val == "THROUGHPUT") {
            return ov::hint::PerformanceMode::THROUGHPUT;
        } else if (val == "CUMULATIVE_THROUGHPUT") {
            return ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT;
        }

        OPENVINO_THROW("Value '", val, "' is not a valid PERFORMANCE_HINT option");
    }

    static std::string toString(const ov::hint::PerformanceMode& val) {
        std::stringstream strStream;
        switch (val) {
        case ov::hint::PerformanceMode::LATENCY:
            strStream << "LATENCY";
            break;
        case ov::hint::PerformanceMode::THROUGHPUT:
            strStream << "THROUGHPUT";
            break;
        case ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT:
            strStream << "CUMULATIVE_THROUGHPUT";
            break;
        default:
            OPENVINO_THROW("Invalid ov::hint::PerformanceMode setting");
            break;
        }
        return strStream.str();
    }
};

struct PERFORMANCE_HINT_NUM_REQUESTS final : OptionBase<PERFORMANCE_HINT_NUM_REQUESTS, uint32_t> {
    static std::string_view key() {
        return ov::hint::num_requests.name();
    }

    /**
     * @brief Returns configuration value if it is valid, otherwise throws
     * @details This is the same function as "InferenceEngine::PerfHintsConfig::CheckPerformanceHintRequestValue",
     * slightly modified as to not rely on the legacy API anymore.
     * @param configuration value as string
     * @return configuration value as number
     */
    static uint32_t parse(std::string_view val) {
        int val_i = -1;
        try {
            val_i = std::stoi(val.data());
            if (val_i >= 0)
                return val_i;
            else
                throw std::logic_error("wrong val");
        } catch (const std::exception&) {
            OPENVINO_THROW("Wrong value of ",
                           val.data(),
                           " for property key ",
                           ov::hint::num_requests.name(),
                           ". Expected only positive integer numbers");
        }
    }

    static uint32_t defaultValue() {
        // Default value depends on PERFORMANCE_HINT, see getOptimalNumberOfInferRequestsInParallel
        // 1 corresponds to LATENCY and default mode (hints not specified)
        return 1u;
    }

    static bool isPublic() {
        return true;
    }

    static ov::PropertyMutability mutability() {
        return ov::PropertyMutability::RW;
    }

    static OptionMode mode() {
        return OptionMode::Both;
    }
};

struct INFERENCE_PRECISION_HINT final : OptionBase<INFERENCE_PRECISION_HINT, ov::element::Type> {
    static std::string_view key() {
        return ov::hint::inference_precision.name();
    }

    static constexpr std::string_view getTypeName() {
        return "ov::hint::inference_precision";
    }

    static ov::element::Type defaultValue() {
        return ov::element::f16;
    }

    static uint32_t compilerSupportVersion() {
        return ONEAPI_MAKE_VERSION(5, 4);
    }

    static bool isPublic() {
        return true;
    }

    static ov::PropertyMutability mutability() {
        return ov::PropertyMutability::RW;
    }

    static OptionMode mode() {
        return OptionMode::Both;
    }

    static ov::element::Type parse(std::string_view val) {
        if (val.empty() || (val == "f16")) {
            return ov::element::f16;
        } else if (val == "i8") {
            return ov::element::i8;
        } else {
            OPENVINO_THROW("Wrong value ",
                           val.data(),
                           " for property key ",
                           ov::hint::inference_precision.name(),
                           ". Supported values: f16, i8");
        }
    };
};

struct PERF_COUNT final : OptionBase<PERF_COUNT, bool> {
    static std::string_view key() {
        return ov::enable_profiling.name();
    }

    static bool defaultValue() {
        return false;
    }

    static bool isPublic() {
        return true;
    }

    static ov::PropertyMutability mutability() {
        return ov::PropertyMutability::RW;
    }

    static OptionMode mode() {
        return OptionMode::Both;
    }
};

struct LOG_LEVEL final : OptionBase<LOG_LEVEL, ov::log::Level> {
    static std::string_view key() {
        return ov::log::level.name();
    }

    static constexpr std::string_view getTypeName() {
        return "ov::log::Level";
    }

    static std::string_view envVar() {
        return "OV_NPU_LOG_LEVEL";
    }

    static OptionMode mode() {
        return OptionMode::Both;
    }

    static ov::log::Level defaultValue() {
#if defined(NPU_PLUGIN_DEVELOPER_BUILD) || !defined(NDEBUG)
        return ov::log::Level::WARNING;
#else
        return ov::log::Level::ERR;
#endif
    }

    static bool isPublic() {
        return true;
    }
};

struct PLATFORM final : OptionBase<PLATFORM, std::string> {
    static std::string_view key() {
        return ov::intel_npu::platform.name();
    }

    static std::string defaultValue() {
        return std::string(ov::intel_npu::Platform::AUTO_DETECT);
    }

#ifdef NPU_PLUGIN_DEVELOPER_BUILD
    static std::string_view envVar() {
        return "IE_NPU_PLATFORM";
    }
#endif

    static bool isPublic() {
        return false;
    }

    static OptionMode mode() {
        return OptionMode::Both;
    }

    static ov::PropertyMutability mutability() {
        return ov::PropertyMutability::RW;
    }
};

struct DEVICE_ID final : OptionBase<DEVICE_ID, std::string> {
    static std::string_view key() {
        return ov::device::id.name();
    }

    static std::string defaultValue() {
        return {};
    }

    static bool isPublic() {
        return true;
    }

    static ov::PropertyMutability mutability() {
        return ov::PropertyMutability::RW;
    }

    static OptionMode mode() {
        return OptionMode::Both;
    }
};

struct CACHE_DIR final : OptionBase<CACHE_DIR, std::string> {
    static std::string_view key() {
        return ov::cache_dir.name();
    }

    static std::string defaultValue() {
        return {};
    }

    static bool isPublic() {
        return true;
    }

    static ov::PropertyMutability mutability() {
        return ov::PropertyMutability::RW;
    }

    static OptionMode mode() {
        return OptionMode::RunTime;
    }
};

struct CACHE_MODE final : OptionBase<CACHE_MODE, ov::CacheMode> {
    static std::string_view key() {
        return ov::cache_mode.name();
    }

    static constexpr std::string_view getTypeName() {
        return "ov::CacheMode";
    }

    static ov::CacheMode defaultValue() {
        return ov::CacheMode::OPTIMIZE_SPEED;
    }

    static bool isPublic() {
        return true;
    }

    static ov::PropertyMutability mutability() {
        return ov::PropertyMutability::RW;
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static uint32_t compilerSupportVersion() {
        return ONEAPI_MAKE_VERSION(7, 21);
    }

    static ov::CacheMode parse(std::string_view val) {
        std::istringstream stringStream = std::istringstream(std::string(val));
        ov::CacheMode cacheMode;
        stringStream >> cacheMode;
        return cacheMode;
    }

    static std::string toString(const ov::CacheMode& val) {
        std::stringstream strStream;
        strStream << val;
        return strStream.str();
    }
};

struct COMPILED_BLOB final : OptionBase<COMPILED_BLOB, ov::Tensor> {
    static std::string_view key() {
        return ov::hint::compiled_blob.name();
    }

    static constexpr std::string_view getTypeName() {
        return "ov::Tensor";
    }

    static ov::Tensor defaultValue() {
        return ov::Tensor();
    }

    static ov::PropertyMutability mutability() {
        return ov::PropertyMutability::RW;
    }

    static OptionMode mode() {
        return OptionMode::RunTime;
    }

    static ov::Tensor parse(std::string_view) {
        // Cannot/shouldn't parse this due to conversion and ownership reasons. The config option is added only to
        // comply with the OV API without inserting multiple workarounds.
        return defaultValue();
    }

    static std::string toString(const ov::Tensor&) {
        return "";
    }
};

struct LOADED_FROM_CACHE final : OptionBase<LOADED_FROM_CACHE, bool> {
    static std::string_view key() {
        return ov::loaded_from_cache.name();
    }

    static bool defaultValue() {
        return false;
    }

    static bool isPublic() {
        return true;
    }

    static OptionMode mode() {
        return OptionMode::RunTime;
    }
};

struct CACHING_PROPERTIES final : OptionBase<CACHING_PROPERTIES, std::string> {
    static std::string_view key() {
        return ov::internal::caching_properties.name();
    }

    static std::string defaultValue() {
        return {};
    }

    static bool isPublic() {
        return false;
    }

    static OptionMode mode() {
        return OptionMode::RunTime;
    }
};

struct INTERNAL_SUPPORTED_PROPERTIES final : OptionBase<INTERNAL_SUPPORTED_PROPERTIES, std::string> {
    static std::string_view key() {
        return ov::internal::supported_properties.name();
    }

    static std::string defaultValue() {
        return {};
    }

    static bool isPublic() {
        return false;
    }

    static OptionMode mode() {
        return OptionMode::Both;
    }
};

struct BATCH_MODE final : OptionBase<BATCH_MODE, ov::intel_npu::BatchMode> {
    static std::string_view key() {
        return ov::intel_npu::batch_mode.name();
    }

    static constexpr std::string_view getTypeName() {
        return "ov::intel_npu::BatchMode";
    }

    static ov::intel_npu::BatchMode defaultValue() {
        return ov::intel_npu::BatchMode::AUTO;
    }

    static uint32_t compilerSupportVersion() {
        return ONEAPI_MAKE_VERSION(5, 5);
    }

    static bool isPublic() {
        return false;
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static ov::intel_npu::BatchMode parse(std::string_view val) {
        if (val == "AUTO") {
            return ov::intel_npu::BatchMode::AUTO;
        } else if (val == "COMPILER") {
            return ov::intel_npu::BatchMode::COMPILER;
        } else if (val == "PLUGIN") {
            return ov::intel_npu::BatchMode::PLUGIN;
        }

        OPENVINO_THROW("Value '", val, "'is not a valid BATCH_MODE option");
    }

    static std::string toString(const ov::intel_npu::BatchMode& val) {
        std::stringstream strStream;

        strStream << val;

        return strStream.str();
    }
};

struct EXCLUSIVE_ASYNC_REQUESTS final : OptionBase<EXCLUSIVE_ASYNC_REQUESTS, bool> {
    static std::string_view key() {
        return ov::internal::exclusive_async_requests.name();
    }

    static bool defaultValue() {
        return false;
    }

    static bool isPublic() {
        return false;
    }

    static ov::PropertyMutability mutability() {
        return ov::PropertyMutability::RW;
    }

    static constexpr std::string_view getTypeName() {
        return "bool";
    }

    static OptionMode mode() {
        return OptionMode::RunTime;
    }
};

struct PROFILING_TYPE final : OptionBase<PROFILING_TYPE, ov::intel_npu::ProfilingType> {
    static std::string_view key() {
        return ov::intel_npu::profiling_type.name();
    }

    static constexpr std::string_view getTypeName() {
        return "ov::intel_npu::ProfilingType";
    }

    static ov::intel_npu::ProfilingType defaultValue() {
        return ov::intel_npu::ProfilingType::MODEL;
    }

    static ov::intel_npu::ProfilingType parse(std::string_view val) {
        if (val == "MODEL") {
            return ov::intel_npu::ProfilingType::MODEL;
        } else if (val == "INFER") {
            return ov::intel_npu::ProfilingType::INFER;
        }

        OPENVINO_THROW("Value '", val, "' is not a valid PROFILING_TYPE option");
    }

    static std::string toString(const ov::intel_npu::ProfilingType& val) {
        std::stringstream strStream;
        if (val == ov::intel_npu::ProfilingType::MODEL) {
            strStream << "MODEL";
        } else if (val == ov::intel_npu::ProfilingType::INFER) {
            strStream << "INFER";
        } else {
            OPENVINO_THROW("No valid string for current PROFILING_TYPE option");
        }

        return strStream.str();
    }

    static OptionMode mode() {
        return OptionMode::RunTime;
    }

    static bool isPublic() {
        return false;
    }

    static ov::PropertyMutability mutability() {
        return ov::PropertyMutability::RW;
    }
};

struct MODEL_PRIORITY final : OptionBase<MODEL_PRIORITY, ov::hint::Priority> {
    static std::string_view key() {
        return ov::hint::model_priority.name();
    }

    static constexpr std::string_view getTypeName() {
        return "ov::hint::Priority";
    }

    static ov::hint::Priority defaultValue() {
        return ov::hint::Priority::MEDIUM;
    }

    static ov::hint::Priority parse(std::string_view val) {
        std::istringstream stringStream = std::istringstream(std::string(val));
        ov::hint::Priority priority;

        stringStream >> priority;

        return priority;
    }

    static std::string toString(const ov::hint::Priority& val) {
        std::ostringstream stringStream;

        stringStream << val;

        return stringStream.str();
    }

    static OptionMode mode() {
        return OptionMode::RunTime;
    }

    static bool isPublic() {
        return true;
    }

    static ov::PropertyMutability mutability() {
        return ov::PropertyMutability::RW;
    }
};

struct CREATE_EXECUTOR final : OptionBase<CREATE_EXECUTOR, int64_t> {
    static std::string_view key() {
        return ov::intel_npu::create_executor.name();
    }

    static int64_t defaultValue() {
        return 1;
    }

#ifdef NPU_PLUGIN_DEVELOPER_BUILD
    static std::string_view envVar() {
        return "IE_NPU_CREATE_EXECUTOR";
    }
#endif

    static bool isPublic() {
        return false;
    }

    static ov::PropertyMutability mutability() {
        return ov::PropertyMutability::RW;
    }

    static OptionMode mode() {
        return OptionMode::RunTime;
    }
};

struct DEFER_WEIGHTS_LOAD final : OptionBase<DEFER_WEIGHTS_LOAD, bool> {
    static std::string_view key() {
        return ov::intel_npu::defer_weights_load.name();
    }
    static int64_t defaultValue() {
        return false;
    }
    static constexpr std::string_view getTypeName() {
        return "bool";
    }
#ifdef NPU_PLUGIN_DEVELOPER_BUILD
    static std::string_view envVar() {
        return "OV_NPU_DEFER_WEIGHTS_LOAD";
    }
#endif
    static bool isPublic() {
        return true;
    }
    static OptionMode mode() {
        return OptionMode::RunTime;
    }
};

struct WEIGHTS_PATH final : OptionBase<WEIGHTS_PATH, std::string> {
    static std::string_view key() {
        return ov::weights_path.name();
    }

    static constexpr std::string_view getTypeName() {
        return "std::string";
    }

    static std::string defaultValue() {
        return "";
    }

    static OptionMode mode() {
        return OptionMode::RunTime;
    }

    static bool isPublic() {
        return true;
    }

    static ov::PropertyMutability mutability() {
        return ov::PropertyMutability::RW;
    }
};

struct NUM_STREAMS final : OptionBase<NUM_STREAMS, ov::streams::Num> {
    static std::string_view key() {
        return ov::num_streams.name();
    }

    static constexpr std::string_view getTypeName() {
        return "ov::streams::Num";
    }

    // The only supported number for currently supported platforms.
    // FIXME: update in the future
    static ov::streams::Num defaultValue() {
        return ov::streams::Num(1);
    }

    static ov::streams::Num parse(std::string_view val) {
        std::istringstream stringStream = std::istringstream(std::string(val));
        ov::streams::Num numberOfStreams;

        stringStream >> numberOfStreams;

        return numberOfStreams;
    }

    static std::string itoString(const ov::streams::Num& val) {
        std::ostringstream stringStream;

        stringStream << val;

        return stringStream.str();
    }

    static void validateValue(const ov::streams::Num& num) {
        if (defaultValue() != num && ov::streams::AUTO != num) {
            throw std::runtime_error("NUM_STREAMS can not be set");
        }
    }

    static OptionMode mode() {
        return OptionMode::RunTime;
    }

    static bool isPublic() {
        return true;
    }

    static ov::PropertyMutability mutability() {
        return ov::PropertyMutability::RO;
    }
};

struct ENABLE_CPU_PINNING final : OptionBase<ENABLE_CPU_PINNING, bool> {
    static std::string_view key() {
        return ov::hint::enable_cpu_pinning.name();
    }

    static bool defaultValue() {
        return false;
    }

    static bool isPublic() {
        return true;
    }

    static ov::PropertyMutability mutability() {
        return ov::PropertyMutability::RW;
    }

    static OptionMode mode() {
        return OptionMode::RunTime;
    }
};

struct WORKLOAD_TYPE final : OptionBase<WORKLOAD_TYPE, ov::WorkloadType> {
    static std::string_view key() {
        return ov::workload_type.name();
    }

    static ov::WorkloadType defaultValue() {
        return ov::WorkloadType::DEFAULT;
    }

    static bool isPublic() {
        return true;
    }

    static constexpr std::string_view getTypeName() {
        return "ov::WorkloadType";
    }

    static OptionMode mode() {
        return OptionMode::RunTime;
    }

    static ov::PropertyMutability mutability() {
        return ov::PropertyMutability::RW;
    }

    static ov::WorkloadType parse(std::string_view val) {
        std::istringstream ss = std::istringstream(std::string(val));
        ov::WorkloadType workloadType;

        ss >> workloadType;

        return workloadType;
    }

    static std::string toString(const ov::WorkloadType& val) {
        std::ostringstream ss;
        ss << val;
        return ss.str();
    }
};

struct TURBO final : OptionBase<TURBO, bool> {
    static std::string_view key() {
        return ov::intel_npu::turbo.name();
    }

    static bool defaultValue() {
        return false;
    }

    static bool isPublic() {
        return true;
    }

    static uint32_t compilerSupportVersion() {
        return ONEAPI_MAKE_VERSION(7, 21);
    }

    static OptionMode mode() {
        return OptionMode::Both;
    }

    static ov::PropertyMutability mutability() {
        return ov::PropertyMutability::RW;
    }
};

struct COMPILER_TYPE final : OptionBase<COMPILER_TYPE, ov::intel_npu::CompilerType> {
    static std::string_view key() {
        return ov::intel_npu::compiler_type.name();
    }

    static constexpr std::string_view getTypeName() {
        return "ov::intel_npu::CompilerType";
    }

    static std::string_view envVar() {
#ifdef NPU_PLUGIN_DEVELOPER_BUILD
        return "IE_NPU_COMPILER_TYPE";
#else
        return "";
#endif
    }

    static ov::intel_npu::CompilerType defaultValue() {
        return ov::intel_npu::CompilerType::DRIVER;
    }

    static ov::intel_npu::CompilerType parse(std::string_view val) {
        if (val == "MLIR") {
            return ov::intel_npu::CompilerType::MLIR;
        } else if (val == "DRIVER") {
            return ov::intel_npu::CompilerType::DRIVER;
        }

        OPENVINO_THROW("Value '", val, "' is not a valid COMPILER_TYPE option");
    }

    static std::string toString(const ov::intel_npu::CompilerType& val) {
        std::stringstream strStream;
        if (val == ov::intel_npu::CompilerType::MLIR) {
            strStream << "MLIR";
        } else if (val == ov::intel_npu::CompilerType::DRIVER) {
            strStream << "DRIVER";
        } else {
            OPENVINO_THROW("No valid string for current COMPILER_TYPE option");
        }

        return strStream.str();
    }

    static OptionMode mode() {
        return OptionMode::RunTime;
    }

    static ov::PropertyMutability mutability() {
        return ov::PropertyMutability::RW;
    }

    static bool isPublic() {
        return false;
    }
};

struct COMPILATION_MODE final : OptionBase<COMPILATION_MODE, std::string> {
    static std::string_view key() {
        return ov::intel_npu::compilation_mode.name();
    }

#ifdef NPU_PLUGIN_DEVELOPER_BUILD
    static std::string_view envVar() {
        return "IE_NPU_COMPILATION_MODE";
    }
#endif

    static std::string defaultValue() {
        return "";
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return false;
    }

    static ov::PropertyMutability mutability() {
        return ov::PropertyMutability::RW;
    }
};

struct EXECUTION_MODE_HINT final : OptionBase<EXECUTION_MODE_HINT, ov::hint::ExecutionMode> {
    static std::string_view key() {
        return ov::hint::execution_mode.name();
    }

    static constexpr std::string_view getTypeName() {
        return "ov::hint::ExecutionMode";
    }

    static ov::hint::ExecutionMode defaultValue() {
        return ov::hint::ExecutionMode::PERFORMANCE;
    }

    static uint32_t compilerSupportVersion() {
        return ONEAPI_MAKE_VERSION(5, 6);
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return true;
    }

    static ov::PropertyMutability mutability() {
        return ov::PropertyMutability::RW;
    }
};

struct DYNAMIC_SHAPE_TO_STATIC final : OptionBase<DYNAMIC_SHAPE_TO_STATIC, bool> {
    static std::string_view key() {
        return ov::intel_npu::dynamic_shape_to_static.name();
    }

#ifdef NPU_PLUGIN_DEVELOPER_BUILD
    static std::string_view envVar() {
        return "IE_NPU_DYNAMIC_SHAPE_TO_STATIC";
    }
#endif

    static bool defaultValue() {
        return false;
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return false;
    }

    static ov::PropertyMutability mutability() {
        return ov::PropertyMutability::RW;
    }
};

struct COMPILATION_MODE_PARAMS final : OptionBase<COMPILATION_MODE_PARAMS, std::string> {
    static std::string_view key() {
        return ov::intel_npu::compilation_mode_params.name();
    }

    static std::string defaultValue() {
        return {};
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return true;
    }

    static ov::PropertyMutability mutability() {
        return ov::PropertyMutability::RW;
    }
};

struct TILES final : OptionBase<TILES, int64_t> {
    static std::string_view key() {
        return ov::intel_npu::tiles.name();
    }

    static std::vector<std::string_view> deprecatedKeys() {
        return {};
    }

    static int64_t defaultValue() {
        return -1;
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return true;
    }

    static uint32_t compilerSupportVersion() {
        return ONEAPI_MAKE_VERSION(5, 4);
    }

    static ov::PropertyMutability mutability() {
        return ov::PropertyMutability::RW;
    }

#ifdef NPU_PLUGIN_DEVELOPER_BUILD
    static std::string_view envVar() {
        return "IE_NPU_TILES";
    }
#endif
};

struct STEPPING final : OptionBase<STEPPING, int64_t> {
    static std::string_view key() {
        return ov::intel_npu::stepping.name();
    }

    static std::vector<std::string_view> deprecatedKeys() {
        return {};
    }

    static int64_t defaultValue() {
        return -1;
    }

    static uint32_t compilerSupportVersion() {
        return ONEAPI_MAKE_VERSION(5, 3);
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return false;
    }

    static ov::PropertyMutability mutability() {
        return ov::PropertyMutability::RW;
    }
};

struct MAX_TILES final : OptionBase<MAX_TILES, int64_t> {
    static std::string_view key() {
        return ov::intel_npu::max_tiles.name();
    }

    static std::vector<std::string_view> deprecatedKeys() {
        return {};
    }

    static uint32_t compilerSupportVersion() {
        return ONEAPI_MAKE_VERSION(5, 3);
    }

    static int64_t defaultValue() {
        return -1;
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return true;
    }

    static ov::PropertyMutability mutability() {
        return ov::PropertyMutability::RW;
    }
};

struct DMA_ENGINES final : OptionBase<DMA_ENGINES, int64_t> {
    static std::string_view key() {
        return ov::intel_npu::dma_engines.name();
    }

    static std::vector<std::string_view> deprecatedKeys() {
        return {};
    }

    static int64_t defaultValue() {
        return -1;
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return false;
    }

    static ov::PropertyMutability mutability() {
        return ov::PropertyMutability::RW;
    }

#ifdef NPU_PLUGIN_DEVELOPER_BUILD
    static std::string_view envVar() {
        return "IE_NPU_DMA_ENGINES";
    }
#endif
};

struct BACKEND_COMPILATION_PARAMS final : OptionBase<BACKEND_COMPILATION_PARAMS, std::string> {
    static std::string_view key() {
        return ov::intel_npu::backend_compilation_params.name();
    }

    static std::string defaultValue() {
        return {};
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return false;
    }

    static ov::PropertyMutability mutability() {
        return ov::PropertyMutability::RW;
    }
};

struct COMPILATION_NUM_THREADS final : OptionBase<COMPILATION_NUM_THREADS, int32_t> {
    static std::string_view key() {
        return ov::compilation_num_threads.name();
    }

    static int32_t defaultValue() {
        return std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
    }

    static void validateValue(const int32_t& num) {
        if (num <= 0) {
            OPENVINO_THROW("ov::compilation_num_threads must be positive int32 value");
        }
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static ov::PropertyMutability mutability() {
        return ov::PropertyMutability::RW;
    }

    static bool isPublic() {
        return true;
    }
};

struct COMPILER_DYNAMIC_QUANTIZATION final : OptionBase<COMPILER_DYNAMIC_QUANTIZATION, bool> {
    static std::string_view key() {
        return ov::intel_npu::compiler_dynamic_quantization.name();
    }

    static bool defaultValue() {
        return false;
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static uint32_t compilerSupportVersion() {
        return ONEAPI_MAKE_VERSION(7, 1);
    }

    static bool isPublic() {
        return true;
    }
};

struct BYPASS_UMD_CACHING final : OptionBase<BYPASS_UMD_CACHING, bool> {
    static std::string_view key() {
        return ov::intel_npu::bypass_umd_caching.name();
    }

    static bool defaultValue() {
        return false;
    }

    static bool isPublic() {
        return true;
    }

    static OptionMode mode() {
        return OptionMode::RunTime;
    }

    static ov::PropertyMutability mutability() {
        return ov::PropertyMutability::RW;
    }
};

struct RUN_INFERENCES_SEQUENTIALLY final : OptionBase<RUN_INFERENCES_SEQUENTIALLY, bool> {
    static std::string_view key() {
        return ov::intel_npu::run_inferences_sequentially.name();
    }

    static bool defaultValue() {
        return false;
    }

    static bool isPublic() {
        return true;
    }

    static OptionMode mode() {
        return OptionMode::RunTime;
    }
};

struct QDQ_OPTIMIZATION final : OptionBase<QDQ_OPTIMIZATION, bool> {
    static std::string_view key() {
        return ov::intel_npu::qdq_optimization.name();
    }

    static bool defaultValue() {
        return false;
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return true;
    }

    static uint32_t compilerSupportVersion() {
        return ONEAPI_MAKE_VERSION(7, 20);
    }
};

struct QDQ_OPTIMIZATION_FORCED final : OptionBase<QDQ_OPTIMIZATION_FORCED, bool> {
    static std::string_view key() {
        return ov::intel_npu::qdq_optimization_forced.name();
    }

    static bool defaultValue() {
        return false;
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static bool isPublic() {
        return true;
    }

    static uint32_t compilerSupportVersion() {
        return ONEAPI_MAKE_VERSION(7, 21);
    }
};

struct DISABLE_VERSION_CHECK final : OptionBase<DISABLE_VERSION_CHECK, bool> {
    static std::string_view key() {
        return ov::intel_npu::disable_version_check.name();
    }

    static bool defaultValue() {
        return false;
    }

#ifdef NPU_PLUGIN_DEVELOPER_BUILD
    static std::string_view envVar() {
        return "OV_NPU_DISABLE_VERSION_CHECK";
    }
#endif

    static OptionMode mode() {
        return OptionMode::RunTime;
    }
};

struct BATCH_COMPILER_MODE_SETTINGS final : OptionBase<BATCH_COMPILER_MODE_SETTINGS, std::string> {
    static std::string_view key() {
        return ov::intel_npu::batch_compiler_mode_settings.name();
    }

    static std::string defaultValue() {
        return {};
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static uint32_t compilerSupportVersion() {
        return ONEAPI_MAKE_VERSION(7, 4);
    }

    static bool isPublic() {
        return false;
    }
};

struct MODEL_PTR final : OptionBase<MODEL_PTR, std::shared_ptr<const ov::Model>> {
    static std::string_view key() {
        return ov::hint::model.name();
    }

    static constexpr std::string_view getTypeName() {
        return "std::shared_ptr<const ov::Model>";
    }

    static std::shared_ptr<const ov::Model> defaultValue() {
        return nullptr;
    }

    static std::shared_ptr<const ov::Model> parse(std::string_view) {
        // Cannot/shouldn't parse this due to conversion and ownership reasons. The config option is added only to
        // comply with the OV API without inserting multiple workarounds.
        return defaultValue();
    }
    static std::string toString(const std::shared_ptr<const ov::Model>& /* unused m*/) {
        return "";
    }

    static OptionMode mode() {
        return OptionMode::RunTime;
    }

    static bool isPublic() {
        return true;
    }

    static ov::PropertyMutability mutability() {
        return ov::PropertyMutability::RW;
    }
};

struct WEIGHTLESS_BLOB final : OptionBase<WEIGHTLESS_BLOB, bool> {
    static std::string_view key() {
        return ov::intel_npu::weightless_blob.name();
    }

    static bool defaultValue() {
        return false;
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }

    static uint32_t compilerSupportVersion() {
        return ONEAPI_MAKE_VERSION(7, 21);
    }
};

struct SEPARATE_WEIGHTS_VERSION final : OptionBase<SEPARATE_WEIGHTS_VERSION, ov::intel_npu::WSVersion> {
    static std::string_view key() {
        return ov::intel_npu::separate_weights_version.name();
    }

    static constexpr std::string_view getTypeName() {
        return "ov::intel_npu::WSVersion";
    }

    static ov::intel_npu::WSVersion defaultValue() {
        return ov::intel_npu::WSVersion::ITERATIVE;
    }

    static ov::intel_npu::WSVersion parse(std::string_view val) {
        std::istringstream stringStream = std::istringstream(std::string(val));
        ov::intel_npu::WSVersion wsVersion;
        stringStream >> wsVersion;
        return wsVersion;
    }

    static std::string toString(const ov::intel_npu::WSVersion& val) {
        std::stringstream strStream;
        strStream << val;
        return strStream.str();
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }
};

struct WS_COMPILE_CALL_NUMBER final : OptionBase<WS_COMPILE_CALL_NUMBER, uint32_t> {
    static std::string_view key() {
        return ov::intel_npu::ws_compile_call_number.name();
    }

    static uint32_t defaultValue() {
        return 0;
    }

    static uint32_t parse(std::string_view val) {
        int val_i = -1;
        try {
            val_i = std::stoi(val.data());
            if (val_i >= 0) {
                return val_i;
            } else {
                throw std::logic_error("wrong val");
            }
        } catch (const std::exception&) {
            OPENVINO_THROW("Wrong value of ",
                           val.data(),
                           " for property key ",
                           ov::intel_npu::ws_compile_call_number.name(),
                           ". Expected only positive integer numbers");
        }
    }

    static OptionMode mode() {
        return OptionMode::CompileTime;
    }
};

}  // namespace intel_npu
