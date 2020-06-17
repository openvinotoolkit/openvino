// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ostream>
#include <details/ie_exception.hpp>
#include "sys/timeb.h"
typedef struct {
    std::string sName;
    std::string sType;  //  if wgt/bias/filt/pwl is writeable, then do not write it to file
    void *pAddress;
    uint32_t nBytes;
} intel_memory_region_t;

typedef unsigned long long time_tsc;


typedef struct
{
    time_tsc            start;      // time value on profiler start
    time_tsc            stop;       // time value on profiler stop
    time_tsc            passed;     // time passed between start and stop
} intel_gna_profiler_tsc;

typedef struct timeb    time_rtc;

typedef struct
{
    time_rtc            start;      // time value on profiler start
    time_rtc            stop;       // time value on profiler stop
    time_rtc            passed;     // time passed between start and stop
} intel_gna_profiler_rtc;

//#define GNA_DEBUG
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




#define THROW_GNA_EXCEPTION THROW_IE_EXCEPTION << "[GNAPlugin] in function " << __PRETTY_FUNCTION__<< ": "
