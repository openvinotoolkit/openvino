// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ze_common.hpp"

namespace cldnn {
namespace ze {

std::shared_ptr<::ov::zero::ZeroApi> ze_api = ::ov::zero::ZeroApi::getInstance();

}  // namespace ze
}  // namespace cldnn
