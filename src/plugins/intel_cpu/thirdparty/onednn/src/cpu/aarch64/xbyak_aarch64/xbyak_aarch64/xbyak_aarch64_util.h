#pragma once
/*******************************************************************************
 * Copyright 2020-2021 FUJITSU LIMITED
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
#ifndef XBYAK_AARCH64_UTIL_H_
#define XBYAK_AARCH64_UTIL_H_

#include <stdint.h>
#ifdef __linux__
#include <sys/auxv.h>
#include <sys/prctl.h>

/* In old Linux such as Ubuntu 16.04, HWCAP_ATOMICS, HWCAP_FP, HWCAP_ASIMD
   can not be found in <bits/hwcap.h> which is included from <sys/auxv.h>.
   Xbyak_aarch64 uses <asm/hwcap.h> as an alternative.
 */
#ifndef HWCAP_FP
#include <asm/hwcap.h>
#endif

#elif defined(__APPLE__)
#include <sys/sysctl.h>
#endif

#include "xbyak_aarch64_err.h"

namespace Xbyak_aarch64 {
namespace util {

enum sveLen_t {
  SVE_NONE = 0,
  SVE_128 = 16 * 1,
  SVE_256 = 16 * 2,
  SVE_384 = 16 * 3,
  SVE_512 = 16 * 4,
  SVE_640 = 16 * 5,
  SVE_768 = 16 * 6,
  SVE_896 = 16 * 7,
  SVE_1024 = 16 * 8,
  SVE_1152 = 16 * 9,
  SVE_1280 = 16 * 10,
  SVE_1408 = 16 * 11,
  SVE_1536 = 16 * 12,
  SVE_1664 = 16 * 13,
  SVE_1792 = 16 * 14,
  SVE_1920 = 16 * 15,
  SVE_2048 = 16 * 16,
};

#ifdef __APPLE__
constexpr char hw_opt_atomics[] = "hw.optional.armv8_1_atomics";
constexpr char hw_opt_fp[] = "hw.optional.floatingpoint";
constexpr char hw_opt_neon[] = "hw.optional.neon";
#endif

/**
   CPU detection class
*/
class Cpu {
  uint64_t type_;
  sveLen_t sveLen_;

public:
  typedef uint64_t Type;

  static const Type tNONE = 0;
  static const Type tADVSIMD = 1 << 1;
  static const Type tFP = 1 << 2;
  static const Type tSVE = 1 << 3;
  static const Type tATOMIC = 1 << 4;

  Cpu() : type_(tNONE), sveLen_(SVE_NONE) {
#ifdef __linux__
    unsigned long hwcap = getauxval(AT_HWCAP);
    if (hwcap & HWCAP_ATOMICS) {
      type_ |= tATOMIC;
    }

    if (hwcap & HWCAP_FP) {
      type_ |= tFP;
    }
    if (hwcap & HWCAP_ASIMD) {
      type_ |= tADVSIMD;
    }
#ifdef HWCAP_SVE
    /* Some old <sys/auxv.h> may not define HWCAP_SVE.
       In that case, SVE is treated as if it were not supported. */
    if (hwcap & HWCAP_SVE) {
      type_ |= tSVE;
      // svcntb(); if arm_sve.h is available
      sveLen_ = (sveLen_t)prctl(51); // PR_SVE_GET_VL
    }
#endif
#elif defined(__APPLE__)
    size_t val = 0;
    size_t len = sizeof(val);

    if (sysctlbyname(hw_opt_atomics, &val, &len, NULL, 0) != 0)
      throw Error(ERR_INTERNAL);
    else
      type_ |= (val == 1) ? tATOMIC : 0;

    if (sysctlbyname(hw_opt_fp, &val, &len, NULL, 0) != 0)
      throw Error(ERR_INTERNAL);
    else
      type_ |= (val == 1) ? tFP : 0;

    if (sysctlbyname(hw_opt_neon, &val, &len, NULL, 0) != 0)
      throw Error(ERR_INTERNAL);
    else
      type_ |= (val == 1) ? tADVSIMD : 0;
#endif
  }

  Type getType() const { return type_; }
  bool has(Type type) const { return (type & type_) != 0; }
  uint64_t getSveLen() const { return sveLen_; }
  bool isAtomicSupported() const { return type_ & tATOMIC; }
};
} // namespace util
} // namespace Xbyak_aarch64
#endif
