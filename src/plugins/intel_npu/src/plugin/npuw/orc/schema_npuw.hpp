// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../orc.hpp"

namespace ov::npuw::orc::schema_npuw {

inline constexpr SchemaUUID NPUW_ORC_PARTITIONED_SCHEMA =
    {0x4E, 0x50, 0x55, 0x57, 0x43, 0x4D, 0x4F, 0x44, 0x50, 0x48, 0x41, 0x53, 0x45, 0x30, 0x30, 0x31};

// Keep the top-level on-wire type IDs centralized so future schemas can reuse
// the same registry without accidental collisions.
enum class PartitionedModel : TypeId {
    ID = 100,
};

enum class Subgraph : TypeId {
    ID = 200,
};

enum class WeightsBank : TypeId {
    ID = 300,
};

}  // namespace ov::npuw::orc::schema_npuw
