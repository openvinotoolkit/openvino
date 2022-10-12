// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace  GNAPluginNS {
/**
 * GNA primitive created in sorting order for this copy layer
 */
static constexpr auto CopyLayerName = "Copy";
/**
 * GNA primitive created at the end of primitives sequence
 */
static constexpr auto DelayedCopyLayerName = "DelayedCopy";

}  // namespace GNAPluginNS
