// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/visibility.hpp"

// Now we use the generic helper definitions above to define INTERPRETER_API
// INTERPRETER_API is used for the public API symbols. It either DLL imports or DLL exports
//    (or does nothing for static build)

#ifdef INTERPRETER_BACKEND_EXPORTS
#    define INTERPRETER_BACKEND_API OPENVINO_CORE_EXPORTS
#else
#    define INTERPRETER_BACKEND_API OPENVINO_CORE_IMPORTS
#endif  // INTERPRETER_DLL_EXPORTS
