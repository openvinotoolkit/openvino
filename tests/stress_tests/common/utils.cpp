#include "utils.h"

#include <string>
#include <string.h>

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


static size_t parseLine(char* line) {
    // This assumes that a digit will be found and the line ends in " Kb".
    size_t i = strlen(line);
    const char* p = line;
    while (*p <'0' || *p > '9') p++;
    line[i-3] = '\0';
    i = (size_t)atoi(p);
    return i;
}

#ifdef _WIN32
size_t getVmSizeInKB() {
                // TODO rewrite for Virtual Memory
                PROCESS_MEMORY_COUNTERS pmc;
                pmc.cb = sizeof(PROCESS_MEMORY_COUNTERS);
                GetProcessMemoryInfo(GetCurrentProcess(),&pmc, pmc.cb);
                return pmc.WorkingSetSize;
	    }
#else
size_t getVirtualMemoryInKB(char *name){
    FILE* file = fopen("/proc/self/status", "r");
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

size_t getVmSizeInKB() {return getVirtualMemoryInKB((char*) "VmSize:");}
size_t getVmPeakInKB() {return getVirtualMemoryInKB((char*) "VmPeak:");}
size_t getVmRSSInKB() {return getVirtualMemoryInKB((char*) "VmRSS:");}
size_t getVmHWMInKB() {return getVirtualMemoryInKB((char*) "VmHWM:");}

#endif
