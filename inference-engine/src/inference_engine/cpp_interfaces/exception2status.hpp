// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

/**
 * \brief Wrappers from c++ function to c-style one
 * \file cpp2c.hpp
 */
#pragma once

#include <string>
#include "description_buffer.hpp"

namespace InferenceEngine {

/**
 * @brief conversion of c++ exceptioned function call into c-style one
 */
#define TO_STATUS(x)\
try {x; return OK;\
} catch (const InferenceEngine::details::InferenceEngineException & iex) { \
    return InferenceEngine::DescriptionBuffer((iex.hasStatus() ? iex.getStatus() : GENERAL_ERROR), \
            resp) << iex.what(); \
} catch (const std::exception & ex) {\
    return InferenceEngine::DescriptionBuffer(GENERAL_ERROR, resp) << ex.what();\
} catch (...) {\
    return InferenceEngine::DescriptionBuffer(UNEXPECTED);\
}

#define TO_STATUS_NO_RESP(x)\
try {x; return OK;\
} catch (const InferenceEngine::details::InferenceEngineException & iex) { \
    return InferenceEngine::DescriptionBuffer(iex.hasStatus() ? iex.getStatus() : GENERAL_ERROR)\
        << iex.what(); \
} catch (const std::exception & ex) {\
    return InferenceEngine::DescriptionBuffer(GENERAL_ERROR) << ex.what();\
} catch (...) {\
    return InferenceEngine::DescriptionBuffer(UNEXPECTED);\
}

#define NO_EXCEPT_CALL_RETURN_VOID(x)\
try { return x; \
} catch (const std::exception & ex) {\
    return;\
}

#define NO_EXCEPT_CALL_RETURN_STATUS(x)\
try {return x;\
} catch (const InferenceEngine::details::InferenceEngineException & iex) { \
    return InferenceEngine::DescriptionBuffer(iex.hasStatus() ? iex.getStatus() : GENERAL_ERROR, \
            resp) << iex.what(); \
} catch (const std::exception & ex) {\
    return InferenceEngine::DescriptionBuffer(GENERAL_ERROR, resp) << ex.what();\
} catch (...) {\
    return InferenceEngine::DescriptionBuffer(UNEXPECTED);\
}

// TODO: replace by hierarchy of exceptions
#define PARAMETER_MISMATCH_str std::string("[PARAMETER_MISMATCH] ")
#define NETWORK_NOT_LOADED_str std::string("[NETWORK_NOT_LOADED] ")
#define NOT_FOUND_str std::string("[NOT_FOUND] ")
#define RESULT_NOT_READY_str  std::string("[RESULT_NOT_READY] ")
#define INFER_NOT_STARTED_str  std::string("[INFER_NOT_STARTED] ")
#define REQUEST_BUSY_str std::string("[REQUEST_BUSY] ")
#define NOT_IMPLEMENTED_str std::string("[NOT_IMPLEMENTED] ")
#define NOT_ALLOCATED_str std::string("[NOT_ALLOCATED] ")

}  // namespace InferenceEngine
