// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief a header that defines helpers for GPU plugin-specific wrappers
 *
 * @file gpu_context_helpers.hpp
 */
#pragma once

#if !defined(IN_OV_COMPONENT) && !defined(IE_LEGACY_HEADER_INCLUDED)
#    define IE_LEGACY_HEADER_INCLUDED
#    ifdef _MSC_VER
#        pragma message( \
            "The Inference Engine API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    else
#        warning("The Inference Engine API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    endif
#endif

#include <string>

#include "ie_parameter.hpp"

IE_SUPPRESS_DEPRECATED_START
namespace InferenceEngine {

namespace gpu {

namespace details {
/**
 * @brief This wrapper class is used to obtain low-level handles
 * from remote blob or context object parameters.
 */
class INFERENCE_ENGINE_1_0_DEPRECATED param_map_obj_getter {
protected:
    /**
     * @brief Template function that returns specified
     * object parameter typecasted to desired user type
     */
    template <typename Result, typename Tmp>
    Result _ObjFromParams(const ParamMap& params,
                          const std::string& handle_Key,
                          const std::string& type_Key,
                          const std::string& obj_T1,
                          const std::string& obj_T2 = "__") const {
        auto itrType = params.find(type_Key);
        if (itrType == params.end())
            IE_THROW() << "Parameter of type " << type_Key << " not found";

        std::string param_val = itrType->second.as<std::string>();
        if (obj_T1 != param_val && obj_T2 != param_val)
            IE_THROW() << "Unexpected object type " << param_val;

        auto itrHandle = params.find(handle_Key);
        if (itrHandle == params.end()) {
            IE_THROW() << "No parameter " << handle_Key << " found";
        }

        return static_cast<Result>(itrHandle->second.as<Tmp>());
    }

    /**
     * @brief Same as _ObjFromParams(), but should be used if check
     * for object type is not required
     */
    template <typename Result>
    Result _ObjFromParamSimple(const ParamMap& params, const std::string& handle_Key) const {
        auto itrHandle = params.find(handle_Key);
        if (itrHandle == params.end()) {
            IE_THROW() << "No parameter " << handle_Key << " found";
        }

        return itrHandle->second.as<Result>();
    }

    /**
     * @brief Template function that extracts string value
     * from map entry under specified key
     */
    std::string _StrFromParams(const ParamMap& params, std::string Key) const {
        auto itrType = params.find(Key);
        if (itrType == params.end())
            IE_THROW() << "Parameter key " << Key << " not found";
        return itrType->second.as<std::string>();
    }
};

}  // namespace details

}  // namespace gpu

}  // namespace InferenceEngine
IE_SUPPRESS_DEPRECATED_END
