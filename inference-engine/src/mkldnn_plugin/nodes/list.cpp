// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "nodes/list.hpp"

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

#define FACTORY_DECLARATION(__prim, __type) \
    void __prim ## __type(MKLDNNExtensions * extInstance)

#define FACTORY_CALL(__prim, __type) \
    __prim ## __type(this)

#define MKLDNN_EXTENSION_NODE(__prim, __type) FACTORY_DECLARATION(__prim, __type)
# include "list_tbl.hpp"
#undef MKLDNN_EXTENSION_NODE

MKLDNNExtensions::MKLDNNExtensions() {
    #define MKLDNN_EXTENSION_NODE(__prim, __type) FACTORY_CALL(__prim, __type)
    # include "list_tbl.hpp"
    #undef MKLDNN_EXTENSION_NODE
}

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
