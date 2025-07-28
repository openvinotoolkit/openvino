// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "nodes/executors/eltwise_config.hpp"
#ifdef CPU_DEBUG_CAPS

#    include <ostream>

#    include "fullyconnected_config.hpp"
#    include "nodes/executors/convolution_config.hpp"
#    include "printers.hpp"

namespace ov::intel_cpu {

std::ostream& operator<<(std::ostream& os, [[maybe_unused]] const FCAttrs& attrs) {
    // @todo print Attrs
    return os;
}

std::ostream& operator<<(std::ostream& os, [[maybe_unused]] const ConvAttrs& attrs) {
    // @todo print Attrs
    return os;
}

std::ostream& operator<<(std::ostream& os, [[maybe_unused]] const EltwiseAttrs& attrs) {
    // @todo print Attrs
    return os;
}

}  // namespace ov::intel_cpu

#endif  // CPU_DEBUG_CAPS
