// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/runtime/core.hpp>

int main() {
    //! [ov:execution_mode:part0]
    ov::Core core;
    // in case of Accuracy
    core.set_property("CPU", ov::hint::execution_mode(ov::hint::ExecutionMode::ACCURACY));
    // in case of Performance
    core.set_property("CPU", ov::hint::execution_mode(ov::hint::ExecutionMode::PERFORMANCE));
    //! [ov:execution_mode:part0]

    return 0;
}
