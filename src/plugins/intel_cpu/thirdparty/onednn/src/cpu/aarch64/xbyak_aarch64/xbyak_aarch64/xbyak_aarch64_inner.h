#pragma once
/*******************************************************************************
 * Copyright 2019-2021 FUJITSU LIMITED
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

enum {
  DEFAULT_MAX_CODE_SIZE = 4096,
  VERSION = 0x5800 /* 0xABCD = A.BC(D) */
};

namespace inner {

inline size_t getPageSize() {
#ifdef __GNUC__
  static const size_t pageSize = sysconf(_SC_PAGESIZE);
#else
  static const size_t pageSize = 4096;
#endif
  return pageSize;
}

inline bool IsInDisp8(uint32_t x) { return 0xFFFFFF80 <= x || x <= 0x7F; }
inline bool IsInInt32(uint64_t x) { return ~uint64_t(0x7fffffffu) <= x || x <= 0x7FFFFFFFU; }

inline uint32_t VerifyInInt32(uint64_t x) {
  if (!IsInInt32(x))
    throw Error(ERR_OFFSET_IS_TOO_BIG);
  return static_cast<uint32_t>(x);
}

constexpr uint32_t genSysInstOp(uint32_t op1, uint32_t CRn, uint32_t CRm, uint32_t op2) { return (op1 << 11) | (CRn << 7) | (CRm << 3) | op2; }

} // namespace inner
