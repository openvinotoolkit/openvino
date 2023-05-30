// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/runtime/core.hpp>

int main() {
    //! [ov:execution_mode:part0]
    ov::Core core;
    core.set_property("CPU", ov::hint::execution_mode(ov::hint::ExecutionMode::ACCURACY));
    //! [ov:execution_mode:part0]

    return 0;
}
