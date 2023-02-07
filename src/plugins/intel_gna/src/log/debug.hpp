// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>

#ifdef GNA_DEBUG

/**
 * @brief used for creating graphviz charts, and layers dump
 */
#    define PLOT

/**
 * @brief used for dumping allocated memory
 */
#    define GNA_MEMORY_DUMP

#endif

/**
 * @brief gna_plugin exception unification
 */
#ifdef __PRETTY_FUNCTION__
#    undef __PRETTY_FUNCTION__
#endif
#ifdef _WIN32
#    define __PRETTY_FUNCTION__ __FUNCSIG__
#else
#    define __PRETTY_FUNCTION__ __FUNCTION__
#endif

#define GNA_LAYER_ASSERT(layer, expr)                      \
    if (!(expr)) {                                         \
        THROW_GNA_LAYER_EXCEPTION(layer) << ": " << #expr; \
    }
#define THROW_GNA_EXCEPTION              IE_THROW() << "[openvino_intel_gna_plugin] in function " << __PRETTY_FUNCTION__ << ": "
#define THROW_GNA_LAYER_EXCEPTION(layer) THROW_GNA_EXCEPTION << LAYER_NAME(layer)
#define LAYER_NAME(layer)                (layer)->type << " layer : \"" << (layer)->name << "\" "
