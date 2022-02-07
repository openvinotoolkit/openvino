// Copyright (C) 2018-2022 Intel Corporation
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
        GetProcessMemoryInfo(GetCurrentProcess(),&pmc, pmc.cb);
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

/// Parses number from provided string
    static int parseLine(std::string line) {
        std::string res = "";
        for (auto c: line)
            if (isdigit(c))
                res += c;
        if (res.empty())
            // If number wasn't found return -1
            return -1;
        return std::stoi(res);
    }

    size_t getSystemDataByName(char *name) {
        FILE *file = fopen("/proc/self/status", "r");
        size_t result = 0;
        if (file != nullptr) {
            char line[128];

            while (fgets(line, 128, file) != NULL) {
                if (strncmp(line, name, strlen(name)) == 0) {
                    result = parseLine(line);
                    break;
                }
            }
            fclose(file);
        }
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
