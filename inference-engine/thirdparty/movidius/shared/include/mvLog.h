/*
* Copyright 2017-2019 Intel Corporation.
* The source code, information and material ("Material") contained herein is
* owned by Intel Corporation or its suppliers or licensors, and title to such
* Material remains with Intel Corporation or its suppliers or licensors.
* The Material contains proprietary information of Intel or its suppliers and
* licensors. The Material is protected by worldwide copyright laws and treaty
* provisions.
* No part of the Material may be used, copied, reproduced, modified, published,
* uploaded, posted, transmitted, distributed or disclosed in any way without
* Intel's prior express written permission. No license under any patent,
* copyright or other intellectual property rights in the Material is granted to
* or conferred upon you, either expressly, by implication, inducement, estoppel
* or otherwise.
* Any license under such intellectual property rights must be express and
* approved by Intel in writing.
*/

/*
 * Add logging capabilities over simple printf.
 * Allows 5 different logging levels:
 *
 * MVLOG_DEBUG = 0
 * MVLOG_INFO = 1
 * MVLOG_WARN = 2
 * MVLOG_ERROR = 3
 * MVLOG_FATAL = 4
 * Before including header, a unit name can be set, otherwise defaults to global. eg:
 *
 * #define MVLOG_UNIT_NAME unitname
 * #include <mvLog.h>
 * Setting log level through debugger can be done in the following way:
 * mset mvLogLevel_unitname 2
 * Will set log level to warnings and above
 */
#ifndef MVLOG_H__
#define MVLOG_H__

#include <stdio.h>
#include <stdarg.h>
#include <inttypes.h>
#include <time.h>

#ifdef __RTEMS__
#include <rtems.h>
#include <rtems/bspIo.h>
#endif

 // Windows-only
#if (defined (WINNT) || defined(_WIN32) || defined(_WIN64) )
#define __attribute__(x)
#define FUNCATTR_WEAK static
#else
#define FUNCATTR_WEAK
#endif

#ifndef MVLOG_UNIT_NAME
#define MVLOG_UNIT_NAME global
#endif

#define _MVLOGLEVEL(UNIT_NAME)  mvLogLevel_ ## UNIT_NAME
#define  MVLOGLEVEL(UNIT_NAME) _MVLOGLEVEL(UNIT_NAME)

#define STR(x) _STR(x)
#define _STR(x)  #x

#define UNIT_NAME_STR STR(MVLOG_UNIT_NAME)

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_WHITE   "\x1b[37m"
#define ANSI_COLOR_RESET   "\x1b[0m"

#ifndef MVLOG_DEBUG_COLOR
#define MVLOG_DEBUG_COLOR ANSI_COLOR_WHITE
#endif

#ifndef MVLOG_INFO_COLOR
#define MVLOG_INFO_COLOR ANSI_COLOR_CYAN
#endif

#ifndef MVLOG_WARN_COLOR
#define MVLOG_WARN_COLOR ANSI_COLOR_YELLOW
#endif

#ifndef MVLOG_ERROR_COLOR
#define MVLOG_ERROR_COLOR ANSI_COLOR_MAGENTA
#endif

#ifndef MVLOG_FATAL_COLOR
#define MVLOG_FATAL_COLOR ANSI_COLOR_RED
#endif

typedef enum mvLog_t{
    MVLOG_DEBUG = 0,
    MVLOG_INFO,
    MVLOG_WARN,
    MVLOG_ERROR,
    MVLOG_FATAL,
    MVLOG_LAST,
} mvLog_t;

static const char mvLogHeader[MVLOG_LAST][30] =
{
    MVLOG_DEBUG_COLOR "D:",
    MVLOG_INFO_COLOR  "I:",
    MVLOG_WARN_COLOR  "W:",
    MVLOG_ERROR_COLOR "E:",
    MVLOG_FATAL_COLOR "F:"
};

FUNCATTR_WEAK unsigned int __attribute__ ((weak)) MVLOGLEVEL(MVLOG_UNIT_NAME) = MVLOG_LAST; // not set by default

FUNCATTR_WEAK unsigned int __attribute__ ((weak)) MVLOGLEVEL(default) = MVLOG_WARN;

static int __attribute__ ((unused))
logprintf(enum mvLog_t lvl, const char * func, const int line,
                     const char * format, ...)
{
    if((MVLOGLEVEL(MVLOG_UNIT_NAME) == MVLOG_LAST && lvl < MVLOGLEVEL(default)))
        return 0;

    if((MVLOGLEVEL(MVLOG_UNIT_NAME) < MVLOG_LAST && lvl < MVLOGLEVEL(MVLOG_UNIT_NAME)))
        return 0;

    const char headerFormat[] = "%s [%s] [%10" PRId64 "] %s:%d\t";
#ifdef __RTEMS__
    uint64_t timestamp = rtems_clock_get_uptime_nanoseconds() / 1000;
#elif !defined(_WIN32)
    struct timespec spec;
    clock_gettime(CLOCK_REALTIME, &spec);
    uint64_t timestamp = (spec.tv_sec % 1000) * 1000 + spec.tv_nsec / 1e6;
#else
    uint64_t timestamp = 0;
#endif
    va_list args;
    va_start (args, format);

#ifdef __RTEMS__
    if(!rtems_interrupt_is_in_progress())
    {
#endif
        fprintf(stdout, headerFormat, mvLogHeader[lvl], UNIT_NAME_STR, timestamp, func, line);
        vfprintf(stdout, format, args);
        fprintf(stdout, "%s\n", ANSI_COLOR_RESET);
#ifdef __RTEMS__
    }
    else
    {
        printk(headerFormat, mvLogHeader[lvl], UNIT_NAME_STR, timestamp, func, line);
        vprintk(format, args);
        printk("%s\n", ANSI_COLOR_RESET);
    }
#endif
    va_end (args);
    return 0;
}

#define mvLog(lvl, format, ...)                                 \
    logprintf(lvl, __func__, __LINE__, format, ##__VA_ARGS__)

// Set log level for the current unit. Note that the level must be smaller than the global default
#define mvLogLevelSet(lvl) if(lvl < MVLOG_LAST){ MVLOGLEVEL(MVLOG_UNIT_NAME) = lvl; }
// Set the global log level. Can be used to prevent modules from hiding messages (enable all of them with a single change)
#define mvLogDefaultLevelSet(lvl) if(lvl < MVLOG_LAST){ MVLOGLEVEL(default) = lvl; }

#endif
