// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "memory_tests_helper/memory_counter.h"
#include <fstream>
#include <memory>
#include <string>
#include <cstring>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#include <tlhelp32.h>
#else

#include <sys/unistd.h>
#include <sys/wait.h>

#endif

#include "statistics_writer.h"

namespace MemoryTest {
#ifdef _WIN32
    static PROCESS_MEMORY_COUNTERS getMemoryInfo() {
        static PROCESS_MEMORY_COUNTERS pmc;
        pmc.cb = sizeof(PROCESS_MEMORY_COUNTERS);
        if (!GetProcessMemoryInfo(GetCurrentProcess(), &pmc, pmc.cb))
            throw std::runtime_error("Can't get system memory values");
        return pmc;
    }

    size_t getVmSizeInKB() {
        return getMemoryInfo().PagefileUsage / 1024;
        }

    size_t getVmPeakInKB() {
        return getMemoryInfo().PeakPagefileUsage / 1024;
        }

    size_t getVmRSSInKB() {
        return getMemoryInfo().WorkingSetSize / 1024;
        }

    size_t getVmHWMInKB() {
        return getMemoryInfo().PeakWorkingSetSize / 1024;
        }

    size_t getThreadsNum() {
        // first determine the id of the current process
        DWORD const  id = GetCurrentProcessId();

        // then get a process list snapshot.
        HANDLE const  snapshot = CreateToolhelp32Snapshot( TH32CS_SNAPALL, 0 );

        // initialize the process entry structure.
        PROCESSENTRY32 entry = { 0 };
        entry.dwSize = sizeof( entry );

        // get the first process info.
        BOOL  ret = true;
        ret = Process32First( snapshot, &entry );
        while( ret && entry.th32ProcessID != id ) {
            ret = Process32Next( snapshot, &entry );
        }
        CloseHandle( snapshot );
        return ret
            ?   entry.cntThreads
            :   -1;
        }

#else

    size_t getSystemDataByName(char* name) {
        auto parseLine = [](std::string line) -> size_t {
            std::string res = "";
            for (auto c : line)
                if (isdigit(c))
                    res += c;
            if (res.empty())
                throw std::runtime_error("Can't get system memory values");
            return std::stoul(res);
        };

        FILE* file = fopen("/proc/self/status", "r");
        size_t result = 0;
        bool status = false;
        if (file != nullptr) {
            char line[128];

            while (fgets(line, 128, file) != NULL) {
                if (strncmp(line, name, strlen(name)) == 0) {
                    result = parseLine(line);
                    status = true;
                    break;
                }
            }
            fclose(file);
        }
        if (!status)
            throw std::runtime_error("Can't get system memory values");
        return result;
    }

    size_t getVmSizeInKB() { return getSystemDataByName((char *) "VmSize:"); }

    size_t getVmPeakInKB() { return getSystemDataByName((char *) "VmPeak:"); }

    size_t getVmRSSInKB() { return getSystemDataByName((char *) "VmRSS:"); }

    size_t getVmHWMInKB() { return getSystemDataByName((char *) "VmHWM:"); }

    size_t getThreadsNum() { return getSystemDataByName((char *) "Threads:"); }

#endif


    MemoryCounter::MemoryCounter(const std::string &mem_counter_name) {
        name = mem_counter_name;
        std::vector<size_t> memory_measurements = {getVmRSSInKB(), getVmHWMInKB(), getVmSizeInKB(),
                                                   getVmPeakInKB(), getThreadsNum()};
        StatisticsWriter::Instance().addMemCounterToStructure({name, memory_measurements});
    }

} // namespace MemoryTest
