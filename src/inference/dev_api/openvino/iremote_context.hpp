// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief OpenVINO Runtime Remote Context API wrapper
 * @file ie_remote_context.hpp
 */
#pragma once

#include <memory>
#include <openvino/runtime/common.hpp>

namespace ov {

class ICore;

class OPENVINO_RUNTIME_API IRemoteContext : public std::enable_shared_from_this<IRemoteContext> {};

}  // namespace ov
