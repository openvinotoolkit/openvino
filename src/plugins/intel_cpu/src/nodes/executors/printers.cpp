// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef CPU_DEBUG_CAPS

#    include "printers.hpp"

#    include <ostream>

#    include "fullyconnected_config.hpp"
#    include "post_ops.hpp"

namespace ov::intel_cpu {

std::ostream& operator<<(std::ostream& os, const FCAttrs& attrs) {
    // @todo print Attrs
    return os;
}

std::ostream& operator<<(std::ostream& os, const PostOps& postOps) {
    // @todo print PostOps
    return os;
}

}  // namespace ov::intel_cpu

#endif  // CPU_DEBUG_CAPS
