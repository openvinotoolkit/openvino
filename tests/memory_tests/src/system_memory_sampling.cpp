
#include "system_memory_sampling.hpp"


#ifdef _WIN32

#include <stdexcept>
#include <windows.h>
#include <tlhelp32.h>
#include <psapi.h>

static PROCESS_MEMORY_COUNTERS getMemoryInfo() {
    static PROCESS_MEMORY_COUNTERS pmc;
    pmc.cb = sizeof(PROCESS_MEMORY_COUNTERS);
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &pmc, pmc.cb))
        throw std::runtime_error("Can't get system memory values");
    return pmc;
}

static size_t getThreadsNum() {
    // first determine the id of the current process
    DWORD const id = GetCurrentProcessId();

    // then get a process list snapshot.
    HANDLE const snapshot = CreateToolhelp32Snapshot( TH32CS_SNAPALL, 0 );

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


SystemMemorySample sampleSystemMemory() {
    SystemMemorySample out;
    auto meminfo = getMemoryInfo();
    out.virtual_size = meminfo.PagefileUsage / 1024;
    out.virtual_peak = meminfo.PeakPagefileUsage / 1024;
    out.resident_size = meminfo.WorkingSetSize / 1024;
    out.resident_peak = meminfo.PeakWorkingSetSize / 1024;
    out.thread_count = (int32_t) getThreadsNum();
    return out;
}

#else

#include <fstream>
#include <string>


SystemMemorySample sampleSystemMemory() {
    SystemMemorySample out;
    std::ifstream file;
    file.open("/proc/self/status");
    std::string line;
    while (true) {
        if (!std::getline(file, line)) {
            break;
        }
        auto delim_pos = line.find(':');
        if (delim_pos == std::string::npos) {
            continue;
        }
        auto prefix = line.substr(0, delim_pos);
        auto value_start = line.find_first_not_of("\t ", delim_pos + 1);
        auto value_end = line.find_first_of("\t ", value_start);
        if (value_start == std::string::npos) {
            continue;
        }
        auto value = line.substr(value_start, value_end - value_start);
        long ivalue = std::atol(value.c_str());
        if (prefix == "VmSize") {
            out.virtual_size = ivalue;
        } else if (prefix == "VmPeak") {
            out.virtual_peak = ivalue;
        } else if (prefix == "VmRSS") {
            out.resident_size = ivalue;
        } else if (prefix == "VmHWM") {
            out.resident_peak = ivalue;
        } else if (prefix == "Threads") {
            out.thread_count = (int32_t) ivalue;
        }
    }
    return out;
}

#endif
