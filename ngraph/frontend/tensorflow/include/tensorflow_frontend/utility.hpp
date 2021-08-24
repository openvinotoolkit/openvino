// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifdef tensorflow_ngraph_frontend_EXPORTS
#define TF_API NGRAPH_HELPER_DLL_EXPORT
#else
#define TF_API NGRAPH_HELPER_DLL_IMPORT
#endif  // paddlepaddle_ngraph_frontend_EXPORTS

namespace tensorflow {
namespace ngraph_bridge {

#define NGRAPH_TF_FE_NOT_IMPLEMENTED                                                        \
    {                                                                                       \
        std::cerr << "[ NOT IMPLEMENTED ] source: " << __FILE__ << ":" << __LINE__ << "\n"; \
        throw "NOT IMPLEMENTED";                                                            \
    }

#if 0
#define NGRAPH_VLOG(I) std::cerr
#else
#define NGRAPH_VLOG(I) std::ostringstream()
#endif

}  // namespace ngraph_bridge
}  // namespace tensorflow

namespace ngraph {
namespace frontend {
namespace tf {

template <typename T>
bool endsWith(const std::basic_string<T>& str, const std::basic_string<T>& suffix) {
    if (str.length() >= suffix.length()) {
        return (0 == str.compare(str.length() - suffix.length(), suffix.length(), suffix));
    }
    return false;
}

}  // namespace tf
}  // namespace frontend
}  // namespace ngraph
