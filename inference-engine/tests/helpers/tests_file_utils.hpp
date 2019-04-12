// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * \brief TODO: short file description
 * \file file_utils.h
 */
#pragma once
#include <string>
#ifdef _WIN32
#define _WINSOCKAPI_
#include <windows.h>
#endif

#ifdef __MACH__
#include <mach/clock.h>
#include <mach/mach.h>
#endif

namespace testing { namespace FileUtils {
#ifdef _WIN32
    /// @brief TODO: description
    const std::string FileSeparator = "\\";
#else
    /// @brief TODO: description
    const std::string FileSeparator = "/";
#endif
    /// @brief TODO: description
    const std::string FileSeparator2 = "/"; // second option

    /**
     * @brief TODO: description
     * @param fileName - TODO: param
     * @return TODO: ret obj
     */
    long long fileSize(const char *fileName);

    /**
     * @brief TODO: description
     * @param f - TODO: param
     * @return TODO: ret obj
     */
    inline long long fileSize(const std::string& f) {
        return fileSize(f.c_str());
    }

    /**
     * @brief TODO: description
     * @param fileName - TODO: param
     * @return TODO: ret obj
     */
    inline bool fileExist(const char *fileName) {
        return fileSize(fileName)>=0;
    }

    /**
     * @brief TODO: description
     * @param fileName - TODO: param
     * @return TODO: ret obj
     */
    inline bool fileExist(const std::string &fileName) {
        return fileExist(fileName.c_str());
    }

    /**
     * @brief TODO: description
     * @param file_name - TODO: param
     * @param buffer - TODO: param
     * @param maxSize - TODO: param
     */
    void readAllFile(const std::string& file_name, void* buffer, size_t maxSize);

    /**
     * @brief TODO: description
     * @param filepath - TODO: param
     * @return TODO: ret obj
     */
    std::string folderOf(const std::string &filepath);

    /**
     * @brief TODO: description
     * @param folder - TODO: param
     * @param file - TODO: param
     * @return TODO: ret obj
     */
    std::string makePath(const std::string& folder, const std::string& file);

    /**
     * @brief TODO: description
     * @param filepath - TODO: param
     * @return TODO: ret obj
     */
    std::string fileNameNoExt(const std::string &filepath);

    /**
     * @brief TODO: description
     * @param filename - TODO: param
     * @return TODO: ret obj
     */
    std::string fileExt(const char* filename);

    /**
     * @brief TODO: description
     * @param filename - TODO: param
     * @return TODO: ret obj
     */
    std::string fileExt(const std::string &filename);

    /**
     * @brief TODO: description
     * @return TODO: please use c++11 chrono module for time operations
     */
    inline long long GetMicroSecTimer() {

    #ifdef _WIN32
        static LARGE_INTEGER Frequency = { 0 };
        LARGE_INTEGER timer;
        if (Frequency.QuadPart==0) QueryPerformanceFrequency(&Frequency);   
        QueryPerformanceCounter(&timer);
        return (timer.QuadPart * 1000000) / Frequency.QuadPart;
    #else
        struct timespec now;
        #ifdef __MACH__
            clock_serv_t cclock;
            mach_timespec_t mts;
            host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
            clock_get_time(cclock, &mts);
            mach_port_deallocate(mach_task_self(), cclock);
            now.tv_sec = mts.tv_sec;
            now.tv_nsec = mts.tv_nsec;
        #else
            clock_gettime(CLOCK_REALTIME, &now);
        #endif
        return now.tv_sec * 1000000L + now.tv_nsec / 1000;
    #endif


    }
}}

