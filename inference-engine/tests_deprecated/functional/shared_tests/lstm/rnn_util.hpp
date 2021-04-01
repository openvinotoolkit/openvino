// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ie_blob.h"

/**
 * Checkers section (Blob::Ptr) -> bool
 */
using Filler = std::function<void(InferenceEngine::Blob::Ptr)>;

/** Fillers conversion */
Filler reverse(const Filler checker, int axis);
Filler permute(const Filler filler, const std::vector<int> order);
Filler concat(const Filler filler1, const Filler filler2, int axis);

/** Some helpful fillers. To use with std::bind() */
void scalar_filler(InferenceEngine::Blob::Ptr blob, InferenceEngine::SizeVector dims, float val);
void vector_filler(InferenceEngine::Blob::Ptr blob, InferenceEngine::SizeVector dims, std::vector<float> val, int axis);

/**
 * Filler section (Blob::Ptr) -> void
 */
using Checker = std::function<bool(InferenceEngine::Blob::Ptr)>;

/** Checker conversion */
Checker negative(const Checker checker);
Checker reverse(const Checker checker, int axis);
Checker permute(const Checker checker, const std::vector<int> order);
Checker concat(const Checker checker1, const Checker checker2, int axis);

/** Some helpful checkers. To use with std::bind() */
bool scalar_checker (InferenceEngine::Blob::Ptr blob, InferenceEngine::SizeVector dims, float val);
bool vector_checker (InferenceEngine::Blob::Ptr blob, InferenceEngine::SizeVector dims, std::vector<float> val, int axis);