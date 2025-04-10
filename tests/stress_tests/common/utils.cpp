// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.h"

#include <string>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#include <tlhelp32.h>
#else
#include <unistd.h>
#include <sys/wait.h>
#endif


std::string OS_PATH_JOIN(std::initializer_list<std::string> list) {
    if (!list.size())
        return "";
    std::string res = *list.begin();
    for (auto it = list.begin() + 1; it != list.end(); it++) {
        res += OS_SEP + *it;
    }
    return res;
}

std::string fileNameNoExt(const std::string &filepath) {
    auto pos = filepath.rfind('.');
    if (pos == std::string::npos) return filepath;
    return filepath.substr(0, pos);
}

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

size_t getVmSizeInKB() {return getSystemDataByName((char*) "VmSize:");}
size_t getVmPeakInKB() {return getSystemDataByName((char*) "VmPeak:");}
size_t getVmRSSInKB() {return getSystemDataByName((char*) "VmRSS:");}
size_t getVmHWMInKB() {return getSystemDataByName((char*) "VmHWM:");}
size_t getThreadsNum() {return getSystemDataByName((char*) "Threads:");}

#endif

int run_in_processes(const int &numprocesses, const std::function<void()> &function) {
#ifdef _WIN32
    // TODO: implement run in separate process by using WinAPI
    function();
    return 0;
#else
    std::vector<pid_t> child_pids(numprocesses);

    for (int i = 0; i < numprocesses; i++) {
        child_pids[i] = fork();
        if (child_pids[i] == 0) {
            function();
            exit(EXIT_SUCCESS);
        }
    }

    int status = 0;
    for (int i = 0; i < numprocesses; i++) {
        int _status = 0;
        waitpid(child_pids[i], &_status, WSTOPPED);
        if (_status) {
            log_err("Process run # " << i << " failed with exitcode " << _status);
            status = _status;
        }
    }
    return status;
#endif
}

void auto_expand_env_vars(std::string &input) {
    const static std::string pattern1 = "${", pattern2 = "}";
    size_t pattern1_pos, pattern2_pos, envvar_start_pos, envvar_finish_pos;
    while ((pattern1_pos = input.find(pattern1)) != std::string::npos) {
        envvar_start_pos = pattern1_pos + pattern1.length();
        if ((pattern2_pos = input.find(pattern2)) != std::string::npos) {
            envvar_finish_pos = pattern2_pos - pattern2.length();
            const std::string envvar_name = input.substr(envvar_start_pos, envvar_finish_pos - envvar_start_pos + 1);
            const char *envvar_val = getenv(envvar_name.c_str());
            if (envvar_val == NULL)
                throw std::logic_error("Expected environment variable " + envvar_name + " is not set.");
            const std::string envvar_val_s(envvar_val);
            input.replace(pattern1_pos, pattern2_pos - pattern1_pos + 1, envvar_val_s);
        }
    }
}
std::string expand_env_vars(const std::string &input) {
    std::string _input = input;
    auto_expand_env_vars(_input);
    return _input;
}
