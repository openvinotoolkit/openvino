// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include "cpu_detector.hpp"

#ifdef ENABLE_MKL_DNN
#ifdef ENABLE_MKL_DNN_JIT
#define XBYAK_NO_OP_NAMES
#define XBYAK_UNDEF_JNL
#include <xbyak_util.h>
#endif
#endif

namespace InferenceEngine {

#ifdef ENABLE_MKL_DNN
#ifdef ENABLE_MKL_DNN_JIT
static Xbyak::util::Cpu cpu;
#endif
#endif

bool with_cpu_x86_sse42() {
#ifdef ENABLE_MKL_DNN
#ifdef ENABLE_MKL_DNN_JIT
    return cpu.has(Xbyak::util::Cpu::tSSE42);
#else
    return false;
#endif
#else
  #if defined(HAVE_SSE)
      return true;
  #else
      return false;
  #endif
#endif
}

}  // namespace InferenceEngine
