// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

namespace MemoryTest {

/** Encapsulate memory measurements.
Object of a class measures memory at start of object's life cycle.
StatisticsWriter adds MemCounter to the memory structure.
*/

class MemoryCounter {
private:
  std::string name;

public:
  /// Constructs MemoryCounter object.
  MemoryCounter(const std::string &mem_counter_name);
};

#define MEMORY_SNAPSHOT(mem_counter_name) MemoryTest::MemoryCounter mem_counter_name(#mem_counter_name);

} // namespace MemoryTest
