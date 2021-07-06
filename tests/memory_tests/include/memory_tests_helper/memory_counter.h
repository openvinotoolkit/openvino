// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

namespace MemoryTest {

/** Encapsulate memory measurements.
Object of a class measures memory at finish of object's life cycle.
When destroyed, reports measurements.
*/
class MemoryCounter {
private:
  std::string name;

public:
  /// Constructs MemoryCounter object.
  MemoryCounter(const std::string &mem_counter_name);

  /// Destructs MemoryCounter object, measures memory values and reports it.
  ~MemoryCounter();
};

#define SCOPED_MEM_COUNTER(mem_counter_name) MemoryTest::MemoryCounter mem_counter_name(#mem_counter_name);

} // namespace MemoryTest