// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/visibility.hpp"

// Now we use the generic helper definitions above to define INTERPRETER_API
// INTERPRETER_API is used for the public API symbols. It either DLL imports or DLL exports
//    (or does nothing for static build)

#ifdef INTERPRETER_BACKEND_EXPORTS // defined if we are building the INTERPRETER DLL (instead of
// using
// it)
#define INTERPRETER_BACKEND_API NGRAPH_HELPER_DLL_EXPORT
#else
#define INTERPRETER_BACKEND_API NGRAPH_HELPER_DLL_IMPORT
#endif // INTERPRETER_DLL_EXPORTS
