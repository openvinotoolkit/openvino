// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace ov {
namespace test {
namespace utils {

enum ov_entity {
    ie_plugin,
    ie_executable_network,
    ie_infer_request,
    ov_plugin,
    ov_compiled_model,
    ov_infer_request
};

}  // namespace utils
}  // namespace test
}  // namespace ov