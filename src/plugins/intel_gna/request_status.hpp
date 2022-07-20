// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace GNAPluginNS {

/**
 * @brief Enum representing status of request
 */
enum class RequestStatus {
    kNone = 0,       /// request was not initialized
    kAborted = 1,    /// request was aborted
    kPending = 2,    /// request was started and is onging
    kCompleted = 3   /// request was completed with success
};

}  // namespace GNAPluginNS
