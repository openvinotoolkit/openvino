// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
