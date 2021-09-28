/*********************************************************************************************************************************************************************************************************************************************************************************************
#   Intel® Single Event API
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

#define INTEL_ITTNOTIFY_API_PRIVATE

#include "ittnotify.h"
#include "ittnotify_config.h"

#ifdef _WIN32
#    define SEA_EXPORT __declspec(dllexport)
#    define _sprintf   sprintf_s
#else
#    define SEA_EXPORT __attribute__((visibility("default")))
#    define _sprintf   sprintf
#endif

namespace sea {
bool IsVerboseMode();
}

#if defined(_WIN32)
#    define VerbosePrint(...)                              \
        {                                                  \
            if (sea::IsVerboseMode()) {                    \
                std::vector<char> buff(1024);              \
                sprintf_s(buff.data(), 1024, __VA_ARGS__); \
                OutputDebugStringA(buff.data());           \
                printf("%s", buff.data());                 \
            }                                              \
        }
#else
#    define VerbosePrint(...)         \
        {                             \
            if (sea::IsVerboseMode()) \
                printf(__VA_ARGS__);  \
        }
#endif

#include <Recorder.h>
#include <sys/stat.h>

#include "TraceEventFormat.h"
#include "Utils.h"

__itt_global* GetITTGlobal();
extern __itt_domain* g_pIntelSEAPIDomain;

namespace sea {
extern std::string g_savepath;
extern uint64_t g_nAutoCut;
#ifdef __linux
bool WriteFTraceTimeSyncMarkers();  // For Driver instrumentation see: http://lwn.net/Articles/379903/
#endif
void InitSEA();
void FillApiList(__itt_api_info* pApiInfo);
void FinitaLaComedia();
void Counter(const __itt_domain* pDomain,
             __itt_string_handle* pName,
             double value,
             __itt_clock_domain* clock_domain = nullptr,
             unsigned long long timestamp = 0);
__itt_clock_domain* clock_domain_create(__itt_get_clock_info_fn fn, void* fn_data);
void SetCutName(const std::string& path);
void SetFolder(const std::string& path);
void SetRing(uint64_t nanoseconds);
const char* GetProcessName(bool bFullPath);
void FixCounter(__itt_counter_info_t* pCounter);
struct SModuleInfo {
    void* base;
    size_t size;
    std::string path;
};
SModuleInfo Fn2Mdl(void* fn);
std::string GetDir(std::string path, const std::string& append = "");
}  // namespace sea

struct SDomainName {
    __itt_domain* pDomain;
    __itt_string_handle* pName;
};

struct ___itt_counter : public __itt_counter_info_t {};

#include <string>
#define USE_PROBES

#ifdef _WIN32
#    include "windows.h"
#elif defined(__linux__)
#    ifndef USE_PROBES
__thread FILE* stdsrc_trace_info_t::pFile = nullptr;
#    endif
#endif

#ifdef _WIN32
#    define UNICODE_AGNOSTIC(name) name##A
inline std::string W2L(const wchar_t* wstr) {
    size_t len = lstrlenW(wstr);
    char* dest = (char*)alloca(len + 2);
    errno_t err = wcstombs_s(&len, dest, len + 1, wstr, len + 1);
    return std::string(dest, dest + len);
}

static_assert(sizeof(__itt_id) == 24, "sizeof(__itt_id) == 24");
static_assert(sizeof(GUID) == 16, "sizeof(GUID) == 16");

union IdCaster {
    __itt_id from;  // d3 is not used, so we fit d1 and d2 into 16 bytes
    GUID to;
};
#else
#    include <cstdio>
#    define _strdup                strdup
#    define UNICODE_AGNOSTIC(name) name
#endif

namespace sea {
__itt_counter UNICODE_AGNOSTIC(counter_create)(const char* name, const char* domain);
__itt_domain* UNICODE_AGNOSTIC(domain_create)(const char* name);
__itt_string_handle* ITTAPI UNICODE_AGNOSTIC(string_handle_create)(const char* name);

enum SEAFeature {
    sfSEA = 0x1,
    sfSystrace = 0x2,
    sfMetricsFrameworkPublisher = 0x4,
    sfMetricsFrameworkConsumer = 0x8,
    sfStack = 0x10,
    sfConcurrencyVisualizer = 0x20,
    sfRemotery = 0x40,
    sfBrofiler = 0x80,
    sfMemStat = 0x100,
    sfMemCounters = 0x200,
    sfRadTelemetry = 0x400,
};

uint64_t GetFeatureSet();
CTraceEventFormat::SRegularFields GetRegularFields(__itt_clock_domain* clock_domain = nullptr,
                                                   unsigned long long timestamp = 0);

struct SThreadRecord;

static const size_t MAX_HANDLERS = 10;

struct STaskDescriptor {
    STaskDescriptor* prev;
    CTraceEventFormat::SRegularFields rf;
    const __itt_domain* pDomain;
    const __itt_string_handle* pName;
    __itt_id id;
    __itt_id parent;
    void* fn;
    struct SCookie {
        void* pCookie;
        void (*Deleter)(void*);
    };
    SCookie cookies[MAX_HANDLERS];

#ifdef TURBO_MODE
    uint64_t nMemCounter;
    double* pDur;
#endif

    ~STaskDescriptor() {
        for (size_t i = 0; i < MAX_HANDLERS; ++i) {
            if (!cookies[i].pCookie)
                continue;
            cookies[i].Deleter(cookies[i].pCookie);
            cookies[i].pCookie = nullptr;
        }
    }
};

struct IHandler {
protected:
    static bool RegisterHandler(IHandler* pHandler);
    size_t m_cookie = ~0x0;
    void SetCookieIndex(size_t cookie) {
        m_cookie = cookie;
    }

    template <class T, class... TArgs>
    T& Cookie(STaskDescriptor& oTask, TArgs&... args) {
        if (!oTask.cookies[m_cookie].pCookie) {
            struct SDeleter {
                static void Deleter(void* ptr) {
                    placement_free(reinterpret_cast<T*>(ptr));
                }
            };
            oTask.cookies[m_cookie] =
                STaskDescriptor::SCookie{placement_new(T)(args...), SDeleter::Deleter};  // consider placement new here!
        }
        return *reinterpret_cast<T*>(oTask.cookies[m_cookie].pCookie);
    }

    const char* GetScope(__itt_scope theScope) {
        static const char* scopes[] = {"unknown", "global", "track_group", "track", "task", "marker"};

        return scopes[theScope];
    }

public:
    struct SData {
        CTraceEventFormat::SRegularFields rf;
        SThreadRecord* pThreadRecord;
        const __itt_domain* pDomain;
        const __itt_id& taskid;
        const __itt_id& parentid;
        const __itt_string_handle* pName;
    };

    template <class T>
    static T* Register(bool bRegister) {
        T* pObject = nullptr;
#ifndef _DEBUG          // register all in debug to discover all problems sooner
        if (bRegister)  // NOLINT
#endif
        {
            pObject = new T();
            if (!RegisterHandler(pObject)) {
                assert(false);
                delete pObject;
                return nullptr;
            }
        }
        return pObject;
    }

    virtual void Init(const CTraceEventFormat::SRegularFields& main) {}
    virtual void TaskBegin(STaskDescriptor& oTask, bool bOverlapped) {}
    virtual void AddArg(STaskDescriptor& oTask, const __itt_string_handle* pKey, const char* data, size_t length) {}
    virtual void AddArg(STaskDescriptor& oTask, const __itt_string_handle* pKey, double value) {}
    virtual void AddRelation(const CTraceEventFormat::SRegularFields& rf,
                             const __itt_domain* pDomain,
                             __itt_id head,
                             __itt_string_handle* relation,
                             __itt_id tail) {}
    virtual void TaskEnd(STaskDescriptor& oTask, const CTraceEventFormat::SRegularFields& rf, bool bOverlapped) {}
    virtual void Marker(const CTraceEventFormat::SRegularFields& rf,
                        const __itt_domain* pDomain,
                        __itt_id id,
                        __itt_string_handle* pName,
                        __itt_scope scope) {}
    virtual void CreateCounter(const __itt_counter& id) {}
    virtual void Counter(const CTraceEventFormat::SRegularFields& rf,
                         const __itt_domain* pDomain,
                         const __itt_string_handle* pName,
                         double value) {}
    virtual void SetThreadName(const CTraceEventFormat::SRegularFields& rf, const char* name) {}
    virtual void Alloc(const CTraceEventFormat::SRegularFields& rf,
                       const void* addr,
                       size_t size,
                       const char* domain,
                       const char* name) {}
    virtual void Free(const CTraceEventFormat::SRegularFields& rf,
                      const void* addr,
                      size_t size,
                      const char* domain,
                      const char* name) {}

    virtual ~IHandler() {}
};

class COverlapped;

struct SThreadRecord {
    std::map<std::string /*domain*/, CRecorder> files;
    bool bRemoveFiles = false;
    __itt_track* pTrack = nullptr;
    SThreadRecord* pNext = nullptr;
    STaskDescriptor* pTask = nullptr;
    COverlapped* pOverlapped = nullptr;
    bool bAllocRecursion = false;
    void* pLastRecorder = nullptr;
    const void* pLastDomain = nullptr;
    int nSpeedupCounter = 0;
#ifdef TURBO_MODE
    uint64_t nMemMoveCounter = 0;  // updated every time memory window moves
#endif                             // TURBO_MODE
};

void TraverseDomains(const std::function<void(___itt_domain&)>& callback);
void TraverseThreadRecords(const std::function<void(SThreadRecord&)>& callback);

void InitDomain(__itt_domain* pDomain);

struct DomainExtra {
    std::string strDomainPath;                // always changed and accessed under lock
    bool bHasDomainPath = false;              // for light check of strDomainPath.empty() without lock
    SThreadRecord* pThreadRecords = nullptr;  // keeping track of thread records for later freeing
    __itt_clock_domain* pClockDomain = nullptr;
    __itt_track_group* pTrackGroup = nullptr;
};

SThreadRecord* GetThreadRecord();

#define CHECKRET(cond, res)                                                                         \
    {                                                                                               \
        if (!(cond)) {                                                                              \
            VerbosePrint("Error: !(%s) at %s, %s:(%d)\n", #cond, __FUNCTION__, __FILE__, __LINE__); \
            return res;                                                                             \
        }                                                                                           \
    }

class CIttLocker {
    __itt_global* m_pGlobal = nullptr;

public:
    CIttLocker();
    ~CIttLocker();
};

#ifdef _WIN32
const uint32_t FilePermissions = _S_IWRITE | _S_IWRITE;  // read by user, write by user
#else
const uint32_t FilePermissions = S_IRWXU | S_IRWXG | S_IRWXO;  // read by all, write by all
#endif

}  // namespace sea
