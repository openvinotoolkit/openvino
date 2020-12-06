// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

// Each reader must be defined in its own namespace to avoid symbol collisions when statically linking everything together
#ifdef IR_READER_V10
#define InferenceEngineIRReader InferenceEngineIRReaderV10
#else
#define InferenceEngineIRReader InferenceEngineIRReaderV7
#endif

namespace InferenceEngine {
namespace details {
} // namespace details
} // namespace InferenceEngine

// We just define the IR reader namespace and InferenceEngine symbols visibles into it
namespace InferenceEngineIRReader {
using namespace InferenceEngine;
namespace details {
using namespace InferenceEngine::details;
} // namespace details
} // namespace InferenceEngineIRReader
