/*********************************************************************************************************************************************************************************************************************************************************************************************
#   Intel(R) Single Event API
#
#   This file is provided under the BSD 3-Clause license.
#   Copyright (c) 2021, Intel Corporation
#   All rights reserved.
#
#   Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#       Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#       Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#       Neither the name of the Intel Corporation nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
#   IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
#   HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
**********************************************************************************************************************************************************************************************************************************************************************************************/

#pragma once
#include <assert.h>
#include <stdint.h>

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <future>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#if defined(__arm__) && !defined(__aarch64__)
#    define ARM32
#endif

#ifdef _WIN32
#    include <windows.h>
#else
#    include <fcntl.h>
#    include <sys/stat.h>
#    include <sys/types.h>
#endif

inline std::string get_environ_value(const std::string& name) {
#ifdef _WIN32
    size_t sz;
    char* v = NULL;
    _dupenv_s(&v, &sz, name.c_str());

    std::string ret = v ? v : "";
    free(v);

    return ret;
#else
    const char* v = getenv(name.c_str());
    return v ? v : "";
#endif
}

#ifdef _WIN32

// there is bug in VS2012 implementation: high_resolution_clock is in fact not high res...
struct SHiResClock {
    typedef uint64_t rep;
    typedef std::nano period;
    typedef std::chrono::duration<rep, period> duration;
    typedef std::chrono::time_point<SHiResClock> time_point;
    static const bool is_steady = true;
    static uint64_t now64() {
        static long long frequency = 0;
        if (!frequency) {
            QueryPerformanceFrequency(reinterpret_cast<LARGE_INTEGER*>(&frequency));
        }

        LARGE_INTEGER count = {};
        QueryPerformanceCounter(&count);
        return static_cast<uint64_t>(static_cast<double>(count.QuadPart) / frequency * static_cast<rep>(period::den));
    }
    static time_point now() {
        return time_point(duration(now64()));
    }
};

namespace sea {
inline uint64_t GetTime() {
    LARGE_INTEGER count = {};
    QueryPerformanceCounter(&count);
    return count.QuadPart;
}
inline uint64_t GetTimeFreq() {
    static LARGE_INTEGER frequency = {};
    if (!frequency.QuadPart) {
        QueryPerformanceFrequency(&frequency);
    }
    return frequency.QuadPart;
}
}  // namespace sea

#else

typedef std::chrono::high_resolution_clock SHiResClock;
namespace sea {
using namespace std::chrono;
inline uint64_t GetTime() {
    return (uint64_t)duration_cast<nanoseconds>(SHiResClock::now().time_since_epoch()).count();
}
inline uint64_t GetTimeFreq() {
    /*
        TODO:
        struct timespec res = {};
        clock_getres(CLOCK_MONOTONIC_RAW, &res);
        uint64_t freq = 1000000000ULL * (uint64_t)res.tv_sec + (uint64_t)res.tv_nsec;
    */
    static uint64_t freq = SHiResClock::period::num / SHiResClock::period::den;
    return freq;
}
}  // namespace sea
#endif

#ifdef _MSC_VER  // std::mutex won't work in static constructors due to MS bug
class CCriticalSection {
    CRITICAL_SECTION m_cs;

public:
    CCriticalSection() {
        InitializeCriticalSection(&m_cs);
    }
    void lock() {
        EnterCriticalSection(&m_cs);
    }
    void unlock() {
        LeaveCriticalSection(&m_cs);
    }
    ~CCriticalSection() {
        DeleteCriticalSection(&m_cs);
    }
};
typedef CCriticalSection TCritSec;
#else
typedef std::recursive_mutex TCritSec;
#endif

#ifdef _MSC_VER
#    define thread_local __declspec(thread)
#else
#    define thread_local __thread
#endif

template <size_t size>
class CPlacementPool {
    static CPlacementPool& GetPool() {
        static thread_local CPlacementPool* pPool = nullptr;
        if (!pPool)
            pPool = new CPlacementPool;
        return *pPool;
    }

    void* AllocMem() {
        if (m_free.size()) {
            void* ptr = m_free.back();
            m_free.pop_back();
            return ptr;
        }
        return malloc(size);
    }

    void FreeMem(void* ptr) {
        m_free.push_back(ptr);
    }

    std::vector<void*> m_free;

public:
    static void* Alloc() {
        return GetPool().AllocMem();
    }

    template <class T>
    static void Free(T* ptr) {
        if (!ptr)
            return;
        ptr->~T();
        return GetPool().FreeMem(ptr);
    }
    ~CPlacementPool() {
        for (void* ptr : m_free) {
            free(ptr);
        }
    }
};

#define placement_new(T) new (CPlacementPool<sizeof(T)>::Alloc()) T
template <class T>
inline void placement_free(T* ptr) {
    CPlacementPool<sizeof(T)>::Free(ptr);
}

class CScope {
protected:
    std::function<void(void)> m_fn;

public:
    CScope(const std::function<void(void)>& fn) : m_fn(fn) {}
    ~CScope() {
        m_fn();
    }
};

const size_t StackSize = 100;
using TStack = void* [StackSize];
size_t GetStack(TStack& stack);
std::string GetStackString();

namespace sea {
void SetGlobalCrashHandler();
}
