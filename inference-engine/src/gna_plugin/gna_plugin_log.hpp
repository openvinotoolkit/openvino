// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <details/ie_exception.hpp>

// #define GNA_DEBUG
#ifdef GNA_DEBUG
/**
 * @brief used for creating graphviz charts, and layers dump
 */
# define PLOT
# define gnalog() std::cout
# define gnawarn() std::cerr
#else

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

/**
 * @brief gna_plugin exception unification
 */
#ifdef __PRETTY_FUNCTION__
#undef __PRETTY_FUNCTION__
#endif
#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
# define __PRETTY_FUNCTION__ __FUNCSIG__
#else
# define __PRETTY_FUNCTION__ __FUNCTION__
#endif


#endif

#define THROW_GNA_EXCEPTION THROW_IE_EXCEPTION << "[GNAPlugin] in function " << __PRETTY_FUNCTION__<< ": "
