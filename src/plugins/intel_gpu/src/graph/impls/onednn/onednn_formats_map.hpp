// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <map>
#include <vector>
#include <oneapi/dnnl/dnnl.hpp>

namespace cldnn {
namespace onednn {

extern const std::map<int, std::vector<dnnl::memory::format_tag>> form_tags_by_ndims;

}  // namespace onednn
}  // namespace cldnn
