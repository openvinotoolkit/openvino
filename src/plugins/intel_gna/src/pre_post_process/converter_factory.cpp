// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "converter_factory.hpp"

#include <ie_system_conf.h>

#include <memory>

namespace ov {
namespace intel_gna {
namespace pre_post_processing {

std::shared_ptr<DataStorageConverter> ConverterFactory::create_converter() {
#ifdef HAVE_AVX2
    if (InferenceEngine::with_cpu_x86_avx2()) {
        return std::make_shared<DataStorageConverterAvx>();
    } else {
        return nullptr;
    }
#else
    return nullptr;
#endif  // HAVE_AVX2
}
}  // namespace pre_post_processing
}  // namespace intel_gna
}  // namespace ov