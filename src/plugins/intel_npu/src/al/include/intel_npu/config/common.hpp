// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_npu/config/config.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/properties.hpp"

namespace intel_npu {

//
// register
//

void registerCommonOptions(OptionsDesc& desc);

//
// PERFORMANCE_HINT
//

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

    static ov::hint::PerformanceMode parse(std::string_view val);
};

// PERFORMANCE_HINT_NUM_REQUESTS
//

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
};

//
// INFERENCE_PRECISION_HINT
//

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

//
// PERF_COUNT
//

struct PERF_COUNT final : OptionBase<PERF_COUNT, bool> {
    static std::string_view key() {
        return ov::enable_profiling.name();
    }

    static bool defaultValue() {
        return false;
    }
};

//
// LOG_LEVEL
//

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

    static ov::log::Level defaultValue() {
#if defined(NPU_PLUGIN_DEVELOPER_BUILD) || !defined(NDEBUG)
        return ov::log::Level::WARNING;
#else
        return ov::log::Level::ERR;
#endif
    }
};

//
// PLATFORM
//

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
};

//
// DEVICE_ID
//

struct DEVICE_ID final : OptionBase<DEVICE_ID, std::string> {
    static std::string_view key() {
        return ov::device::id.name();
    }

    static std::string defaultValue() {
        return {};
    }
};

//
// CACHE_DIR
//

struct CACHE_DIR final : OptionBase<CACHE_DIR, std::string> {
    static std::string_view key() {
        return ov::cache_dir.name();
    }

    static std::string defaultValue() {
        return {};
    }
};

//
// LOADED_FROM_CACHE
//

struct LOADED_FROM_CACHE final : OptionBase<LOADED_FROM_CACHE, bool> {
    static std::string_view key() {
        return ov::loaded_from_cache.name();
    }

    static bool defaultValue() {
        return false;
    }
};

//
// CACHING PROPERTIES
//

struct CACHING_PROPERTIES final : OptionBase<CACHING_PROPERTIES, std::string> {
    static std::string_view key() {
        return ov::internal::caching_properties.name();
    }

    static std::string defaultValue() {
        return {};
    }
};

//
// INTERNAL SUPPORTED PROPERTIES
//

struct INTERNAL_SUPPORTED_PROPERTIES final : OptionBase<INTERNAL_SUPPORTED_PROPERTIES, std::string> {
    static std::string_view key() {
        return ov::internal::supported_properties.name();
    }

    static std::string defaultValue() {
        return {};
    }
};

//
// BATCH_MODE
//
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

    static ov::intel_npu::BatchMode parse(std::string_view val);

    static std::string toString(const ov::intel_npu::BatchMode& val);
};

}  // namespace intel_npu

namespace ov {
namespace hint {

std::string_view stringifyEnum(PerformanceMode val);

}  // namespace hint
}  // namespace ov
