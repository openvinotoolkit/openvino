// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ostream>
#include <details/ie_exception.hpp>

// #define GNA_DEBUG
#ifdef  GNA_DEBUG
#include <iostream>
/**
 * @brief used for creating graphviz charts, and layers dump
 */
# define PLOT
# define gnalog() std::cout
# define gnawarn() std::cerr
#else

#ifdef VERBOSE
#define VERBOSE_LEVEL (1)
#else
#define VERBOSE_LEVEL (0)
#endif

#ifdef PLOT
#define PLOT_LEVEL (1)
#else
#define PLOT_LEVEL (0)
#endif

class GnaLog {
 public :
    template <class T>
    GnaLog & operator << (const T &obj) {
        return *this;
    }

    GnaLog &  operator<< (std::ostream & (*manip)(std::ostream &)) {
        return *this;
    }
};

inline GnaLog & gnalog() {
    static GnaLog l;
    return l;
}
inline GnaLog & gnawarn() {
    return gnalog();
}

#endif

/**
 * @brief gna_plugin exception unification
 */
#ifdef __PRETTY_FUNCTION__
#undef __PRETTY_FUNCTION__
#endif
#ifdef _WIN32
# define __PRETTY_FUNCTION__ __FUNCSIG__
#else
# define __PRETTY_FUNCTION__ __FUNCTION__
#endif



#define GNA_LAYER_ASSERT(layer, expr)\
if (!(expr)) { \
    THROW_GNA_LAYER_EXCEPTION(layer) << ": " << #expr; \
}
#define THROW_GNA_EXCEPTION THROW_IE_EXCEPTION << "[GNAPlugin] in function " << __PRETTY_FUNCTION__<< ": "
#define THROW_GNA_LAYER_EXCEPTION(layer) THROW_GNA_EXCEPTION << LAYER_NAME(layer)
#define LAYER_NAME(layer) (layer)->type << " layer : \"" << (layer)->name << "\" "

