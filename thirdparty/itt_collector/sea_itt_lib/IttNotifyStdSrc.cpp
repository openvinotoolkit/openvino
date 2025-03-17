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

#include "IttNotifyStdSrc.h"

#include <fcntl.h>
#include <string.h>
#include <sys/types.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <set>
#include <stack>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#    include <direct.h>
#    include <io.h>
#else
#    include <libgen.h>
#    include <pthread.h>
#endif

#ifdef __APPLE__
//#define __APPLE_API_UNSTABLE
#    include <sys/kdebug.h>
#    include <sys/kdebug_signpost.h>
#endif

namespace sea {
IHandler* g_handlers[MAX_HANDLERS] = {};  // 10 is more than enough for now

CIttLocker::CIttLocker() {
    m_pGlobal = GetITTGlobal();
    __itt_mutex_lock(&m_pGlobal->mutex);
}

CIttLocker::~CIttLocker() {
    if (m_pGlobal) {
        __itt_mutex_unlock(&m_pGlobal->mutex);
    }
}

}  // namespace sea

// FIXME: in general add much more comments

std::map<std::string, size_t> g_stats;  // can't be static function variable due to lifetime limits

class CIttFnStat {
public:
    CIttFnStat(const char* name) {
        if (!sea::IsVerboseMode())
            return;
        sea::CIttLocker locker;
        ++GetStats()[name];
    }

    static std::map<std::string, size_t>& GetStats() {
        return g_stats;
    }
};

#ifdef _DEBUG
#    define ITT_FUNCTION_STAT() CIttFnStat oIttFnStat(__FUNCTION__)
#else
#    define ITT_FUNCTION_STAT()
#endif

struct __itt_frame_t {
    __itt_domain* pDomain;
    __itt_id id;
};

inline bool operator<(const __itt_id& left, const __itt_id& right) {
    return memcmp(&left, &right, sizeof(__itt_id)) < 0;
}

inline bool operator==(const __itt_id& left, const __itt_id& right) {
    return (left.d1 == right.d1) && (left.d2 == right.d2);
}

namespace sea {

uint64_t g_nRingBuffer = 1000000000ll * atoi(get_environ_value("INTEL_SEA_RING").c_str());    // in nanoseconds
uint64_t g_nAutoCut = 1024ull * 1024 * atoi(get_environ_value("INTEL_SEA_AUTOCUT").c_str());  // in MB
uint64_t g_features = sea::GetFeatureSet();

class DomainFilter {
protected:
    std::string m_path;
    typedef std::map<std::string /*domain*/, bool /*disabled*/> TDomains;
    TDomains m_domains;

    void ReadFilters(TDomains& domains) {
        std::ifstream ifs(m_path);
        for (std::string domain; std::getline(ifs, domain);) {
            if (domain[0] == '#')
                m_domains[domain.c_str() + 1] = true;
            else
                m_domains[domain] = false;
        }
    }

public:
    DomainFilter() {
        m_path = get_environ_value("INTEL_SEA_FILTER");
        if (m_path.empty())
            return;
        ReadFilters(m_domains);
    }

    operator bool() const {
        return !m_path.empty();
    }

    bool IsEnabled(const char* szDomain) {
        return !m_domains[szDomain];  // new domain gets initialized with bool() which is false, so we invert it
    }

    void Finish() {
        if (m_path.empty())
            return;
        TDomains domains;
        ReadFilters(domains);
        domains.insert(m_domains.begin(), m_domains.end());
        m_domains.swap(domains);

        std::ofstream ifs(m_path);
        for (const auto& pair : m_domains) {
            if (pair.second)
                ifs << '#';
            ifs << pair.first << std::endl;
        }
    }
} g_oDomainFilter;

bool PathExists(const std::string& path) {
#ifdef _WIN32
    return -1 != _access(path.c_str(), 0);
#else
    return -1 != access(path.c_str(), F_OK);
#endif
}

int mkpath(const char* path, uint32_t mode) {
    struct stat sb = {};

    if (!stat(path, &sb))
        return 0;

    char parent[1024] = {};
#ifdef _WIN32
    strcpy_s(parent, path);
#else
    strcpy(parent, path);
#endif
    char* last_slash = strrchr(parent, '//');
    if (!last_slash) {
        VerbosePrint("Invalid dir: %s\n", parent);
        return -1;
    }
    *last_slash = 0;

    int res = mkpath(parent, mode);
    if (res == -1) {
        VerbosePrint("Failed to create dir: %s err=%d\n", parent, errno);
        return res;
    } else {
        VerbosePrint("Created dir: %s\n", parent);
    }

#ifdef _WIN32
    return _mkdir(path);
#else
    return mkdir(path, mode);
#endif
}

std::string GetDir(std::string path, const std::string& append) {
    if (path.empty())
        return path;
    path += append;
    VerbosePrint("GetDir: %s\n", path.c_str());

    std::replace(path.begin(), path.end(), '\\', '/');
    char lastSym = path[path.size() - 1];
    if (lastSym != '/')
        path += "/";

    std::string dir_name = path.substr(0, path.length() - 1);
    mkpath(dir_name.c_str(), FilePermissions);
    return path;
}

std::string GetSavePath() {
    static std::string save_to = get_environ_value("INTEL_SEA_SAVE_TO");
    VerbosePrint("Got save path: %s\n", save_to.c_str());
    if (save_to.empty()) {
        return save_to;
    }
    return GetDir(save_to, ("-" + std::to_string(CTraceEventFormat::GetRegularFields().pid)));
}

bool IsVerboseMode() {
    static bool bVerboseMode = !get_environ_value("INTEL_SEA_VERBOSE").empty();
    return bVerboseMode;
}

std::string g_savepath = GetSavePath();
std::shared_ptr<std::string> g_spCutName;

std::string Escape4Path(std::string str) {
    std::replace_if(
        str.begin(),
        str.end(),
        [](char sym) {
            return strchr("/\\:*?\"<>|", sym);
        },
        '_');
    return str;
}

void InitDomain(__itt_domain* pDomain) {
    CIttLocker locker;
    pDomain->extra2 = new DomainExtra{};
    if (g_savepath.size()) {
        DomainExtra* pDomainExtra = reinterpret_cast<DomainExtra*>(pDomain->extra2);
        pDomainExtra->strDomainPath = GetDir(g_savepath, Escape4Path(pDomain->nameA));
        pDomainExtra->bHasDomainPath = !pDomainExtra->strDomainPath.empty();
    }

    if (!g_oDomainFilter)
        return;
    pDomain->flags = g_oDomainFilter.IsEnabled(pDomain->nameA) ? 1 : 0;
}

SThreadRecord* GetThreadRecord() {
    static thread_local SThreadRecord* pThreadRecord = nullptr;
    if (pThreadRecord)
        return pThreadRecord;

    CIttLocker lock;

    pThreadRecord = new SThreadRecord{};
    static __itt_global* pGlobal = GetITTGlobal();

    __itt_domain* pDomain = pGlobal->domain_list;
    DomainExtra* pDomainExtra = reinterpret_cast<DomainExtra*>(pDomain->extra2);
    SThreadRecord* pRecord = pDomainExtra->pThreadRecords;
    if (pRecord) {
        while (pRecord->pNext)
            pRecord = pRecord->pNext;
        pRecord->pNext = pThreadRecord;
    } else {
        pDomainExtra->pThreadRecords = pThreadRecord;
    }

    return pThreadRecord;
}

void UNICODE_AGNOSTIC(thread_set_name)(const char* name) {
    ITT_FUNCTION_STAT();

    for (size_t i = 0; (i < MAX_HANDLERS) && g_handlers[i]; ++i) {
        g_handlers[i]->SetThreadName(GetRegularFields(), name);
    }

#if defined(__APPLE__)
    pthread_setname_np(name);
#elif defined(__linux__)
    pthread_setname_np(pthread_self(), name);
#endif
}
#ifdef _WIN32
void thread_set_nameW(const wchar_t* name) {
    UNICODE_AGNOSTIC(thread_set_name)(W2L(name).c_str());
}
#endif

inline uint64_t ConvertClockDomains(unsigned long long timestamp, __itt_clock_domain* pClock) {
    if (!pClock)
        return timestamp;
    uint64_t start = *(uint64_t*)pClock->extra2;
    return start + (timestamp - pClock->info.clock_base) * SHiResClock::period::den / pClock->info.clock_freq;
}

CTraceEventFormat::SRegularFields GetRegularFields(__itt_clock_domain* clock_domain, unsigned long long timestamp) {
    CTraceEventFormat::SRegularFields rf = CTraceEventFormat::GetRegularFields();

    __itt_track* pTrack = GetThreadRecord()->pTrack;

    if (pTrack) {
        CTraceEventFormat::SRegularFields& trackRF = *(CTraceEventFormat::SRegularFields*)pTrack->extra2;
        rf.changed |= (rf.pid != trackRF.pid) ? CTraceEventFormat::SRegularFields::ecPid
                                              : CTraceEventFormat::SRegularFields::ecNothing;
        rf.pid = trackRF.pid;
        rf.changed |= (rf.tid != trackRF.tid) ? CTraceEventFormat::SRegularFields::ecTid
                                              : CTraceEventFormat::SRegularFields::ecNothing;
        rf.tid = trackRF.tid;
    }
    if (clock_domain || timestamp) {
        rf.nanoseconds = ConvertClockDomains(timestamp, clock_domain);
        rf.changed |= CTraceEventFormat::SRegularFields::ecTime;
    }
    return rf;
}

__itt_domain* UNICODE_AGNOSTIC(domain_create)(const char* name) {
    ITT_FUNCTION_STAT();
    __itt_domain *h_tail = NULL, *h = NULL;

    if (name == NULL) {
        return NULL;
    }
    {
        CIttLocker locker;
        static __itt_global* pGlobal = GetITTGlobal();
        for (h_tail = NULL, h = pGlobal->domain_list; h != NULL; h_tail = h, h = h->next) {
            if (h->nameA != NULL && !__itt_fstrcmp(h->nameA, name))
                break;
        }
        if (h == NULL) {
            NEW_DOMAIN_A(pGlobal, h, h_tail, name);
        }
    }
    InitDomain(h);
    return h;
}

#ifdef _WIN32
__itt_domain* domain_createW(const wchar_t* name) {
    return UNICODE_AGNOSTIC(domain_create)(W2L(name).c_str());
}
#endif

inline __itt_string_handle* get_tail_of_global_string_list(const __itt_global* const pGlobal) {
    if (!pGlobal->string_list)
        return nullptr;

    __itt_string_handle* result = pGlobal->string_list;

    while (result->next) {
        result = result->next;
    }

    return result;
}

inline __itt_string_handle* create_and_add_string_handle_to_list(const char* name) {
    static __itt_global* pGlobal = GetITTGlobal();
    static __itt_string_handle* string_handle_list_tail = get_tail_of_global_string_list(pGlobal);

    __itt_string_handle* result = NULL;

    NEW_STRING_HANDLE_A(pGlobal, result, string_handle_list_tail, name);
    string_handle_list_tail = result;
    return result;
}

__itt_string_handle* ITTAPI UNICODE_AGNOSTIC(string_handle_create)(const char* name) {
    ITT_FUNCTION_STAT();
    if (name == NULL) {
        return NULL;
    }
    CIttLocker locker;
    static std::unordered_map<std::string, __itt_string_handle*> handle_map;
    auto found_handle = handle_map.find(name);
    if (found_handle != handle_map.end()) {
        return found_handle->second;
    }

    __itt_string_handle* result = create_and_add_string_handle_to_list(name);
    handle_map[name] = result;
    sea::ReportString(result);
    return result;
}

#ifdef _WIN32
__itt_string_handle* string_handle_createW(const wchar_t* name) {
    return UNICODE_AGNOSTIC(string_handle_create)(W2L(name).c_str());
}
#endif

void marker_ex(const __itt_domain* pDomain,
               __itt_clock_domain* clock_domain,
               unsigned long long timestamp,
               __itt_id id,
               __itt_string_handle* pName,
               __itt_scope scope) {
    ITT_FUNCTION_STAT();
    CTraceEventFormat::SRegularFields rf = GetRegularFields(clock_domain, timestamp);

    for (size_t i = 0; (i < MAX_HANDLERS) && g_handlers[i]; ++i) {
        g_handlers[i]->Marker(rf, pDomain, id, pName, scope);
    }
}

void marker(const __itt_domain* pDomain, __itt_id id, __itt_string_handle* pName, __itt_scope scope) {
    ITT_FUNCTION_STAT();
    marker_ex(pDomain, nullptr, 0, id, pName, scope);
}

bool IHandler::RegisterHandler(IHandler* pHandler) {
    for (size_t i = 0; i < MAX_HANDLERS; ++i) {
        if (!g_handlers[i]) {
            g_handlers[i] = pHandler;
            pHandler->SetCookieIndex(i);
            return true;
        }
    }
    return false;
}

// FIXME: Use one coding style, since itt functions are mapped, there's no problem with that
void task_begin(const __itt_domain* pDomain, __itt_id taskid, __itt_id parentid, __itt_string_handle* pName) {
    ITT_FUNCTION_STAT();
    SThreadRecord* pThreadRecord = GetThreadRecord();

    CTraceEventFormat::SRegularFields rf = GetRegularFields();
    pThreadRecord->pTask = placement_new(STaskDescriptor){pThreadRecord->pTask,  // chaining the previous task inside
                                                          rf,
                                                          pDomain,
                                                          pName,
                                                          taskid,
                                                          parentid};

    for (size_t i = 0; (i < MAX_HANDLERS) && g_handlers[i]; ++i) {
        g_handlers[i]->TaskBegin(*pThreadRecord->pTask, false);
    }
}

void task_begin_fn(const __itt_domain* pDomain, __itt_id taskid, __itt_id parentid, void* fn) {
    ITT_FUNCTION_STAT();

    CTraceEventFormat::SRegularFields rf = GetRegularFields();
    SThreadRecord* pThreadRecord = GetThreadRecord();

    pThreadRecord->pTask = placement_new(STaskDescriptor){pThreadRecord->pTask,  // chaining the previous task inside
                                                          rf,
                                                          pDomain,
                                                          nullptr,
                                                          taskid,
                                                          parentid,
                                                          fn};

    for (size_t i = 0; (i < MAX_HANDLERS) && g_handlers[i]; ++i) {
        g_handlers[i]->TaskBegin(*pThreadRecord->pTask, false);
    }
}

void task_end(const __itt_domain* pDomain) {
    ITT_FUNCTION_STAT();

    SThreadRecord* pThreadRecord = GetThreadRecord();
    const char* domain = pDomain->nameA;
    if (!pThreadRecord->pTask) {
        VerbosePrint("Uneven begin/end count for domain: %s\n", domain);
        return;
    }

    CTraceEventFormat::SRegularFields rf = GetRegularFields();  // FIXME: get from begin except for time

    for (size_t i = 0; (i < MAX_HANDLERS) && g_handlers[i]; ++i) {
        g_handlers[i]->TaskEnd(*pThreadRecord->pTask, rf, false);
    }

    STaskDescriptor* prev = pThreadRecord->pTask->prev;
    placement_free(pThreadRecord->pTask);
    pThreadRecord->pTask = prev;
}

void Counter(const __itt_domain* pDomain,
             __itt_string_handle* pName,
             double value,
             __itt_clock_domain* clock_domain,
             unsigned long long timestamp) {
    CTraceEventFormat::SRegularFields rf = GetRegularFields(clock_domain, timestamp);

    for (size_t i = 0; (i < MAX_HANDLERS) && g_handlers[i]; ++i) {
        g_handlers[i]->Counter(rf, pDomain, pName, value);
    }
}

void counter_inc_delta_v3(const __itt_domain* pDomain, __itt_string_handle* pName, unsigned long long delta) {
    ITT_FUNCTION_STAT();
    Counter(pDomain, pName, double(delta));
}

void FixCounter(__itt_counter_info_t* pCounter) {
    pCounter->extra2 = new SDomainName{UNICODE_AGNOSTIC(domain_create)(pCounter->domainA),
                                       UNICODE_AGNOSTIC(string_handle_create)(pCounter->nameA)};
    for (size_t i = 0; (i < MAX_HANDLERS) && g_handlers[i]; ++i) {
        g_handlers[i]->CreateCounter(reinterpret_cast<__itt_counter>(pCounter));
    }
}

__itt_counter ITTAPI UNICODE_AGNOSTIC(counter_create_typed)(const char* name,
                                                            const char* domain,
                                                            __itt_metadata_type type) {
    ITT_FUNCTION_STAT();

    if (!name || !domain)
        return nullptr;

    VerbosePrint("%s: name=%s domain=%s type=%d\n", __FUNCTION__, name, domain, (int)type);

    __itt_counter_info_t *h_tail = NULL, *h = NULL;

    CIttLocker locker;
    __itt_global* pGlobal = GetITTGlobal();
    for (h_tail = NULL, h = pGlobal->counter_list; h != NULL; h_tail = h, h = h->next) {
        if (h->nameA != NULL && h->type == type && !__itt_fstrcmp(h->nameA, name) &&
            ((h->domainA == NULL && domain == NULL) ||
             (h->domainA != NULL && domain != NULL && !__itt_fstrcmp(h->domainA, domain))))
            break;
    }
    if (!h) {
        NEW_COUNTER_A(pGlobal, h, h_tail, name, domain, type);
        FixCounter(h);
    }

    return (__itt_counter)h;
}

#ifdef _WIN32
__itt_counter counter_create_typedW(const wchar_t* name, const wchar_t* domain, __itt_metadata_type type) {
    return UNICODE_AGNOSTIC(counter_create_typed)(W2L(name).c_str(), W2L(domain).c_str(), type);
}
#endif

__itt_counter UNICODE_AGNOSTIC(counter_create)(const char* name, const char* domain) {
    ITT_FUNCTION_STAT();
    return UNICODE_AGNOSTIC(counter_create_typed)(name, domain, __itt_metadata_double);
}

#ifdef _WIN32
__itt_counter counter_createW(const wchar_t* name, const wchar_t* domain) {
    return UNICODE_AGNOSTIC(counter_create)(W2L(name).c_str(), W2L(domain).c_str());
}
#endif

template <class T>
double Convert(void* ptr) {
    return static_cast<double>(*reinterpret_cast<T*>(ptr));
}
typedef double (*FConvert)(void* ptr);

FConvert g_MetatypeFormatConverter[] = {
    nullptr,
    Convert<uint64_t>,
    Convert<int64_t>,
    Convert<uint32_t>,
    Convert<int32_t>,
    Convert<uint16_t>,
    Convert<int16_t>,
    Convert<float>,
    Convert<double>,
};

void counter_set_value_ex(__itt_counter id,
                          __itt_clock_domain* clock_domain,
                          unsigned long long timestamp,
                          void* value_ptr) {
    ITT_FUNCTION_STAT();
    if (id->type < __itt_metadata_u64 || id->type > __itt_metadata_double) {
        VerbosePrint("%s: weird type: %d stack: %s\n", __FUNCTION__, (int)id->type, GetStackString().c_str());
        return;
    }
    double val = g_MetatypeFormatConverter[id->type](value_ptr);
    SDomainName* pDomainName = reinterpret_cast<SDomainName*>(id->extra2);
    Counter(pDomainName->pDomain, pDomainName->pName, val, clock_domain, timestamp);
}

void counter_set_value(__itt_counter id, void* value_ptr) {
    ITT_FUNCTION_STAT();
    counter_set_value_ex(id, nullptr, 0, value_ptr);
}

void UNICODE_AGNOSTIC(sync_create)(void* addr, const char* objtype, const char* objname, int attribute) {
    ITT_FUNCTION_STAT();

    std::string name((attribute == __itt_attr_mutex) ? "mutex:" : "barrier:");
    name += objtype;
    name += ":";
    name += objname;
    __itt_string_handle* pName = UNICODE_AGNOSTIC(string_handle_create)(name.c_str());
    __itt_id id = __itt_id_make(addr, 0);

    CTraceEventFormat::SRegularFields rf = GetRegularFields();
    WriteRecord(ERecordType::ObjectNew, SRecord{rf, *g_pIntelSEAPIDomain, id, __itt_null, pName});
}

#ifdef _WIN32
void sync_createW(void* addr, const wchar_t* objtype, const wchar_t* objname, int attribute) {
    UNICODE_AGNOSTIC(sync_create)(addr, W2L(objtype).c_str(), W2L(objname).c_str(), attribute);
}
#endif

void sync_destroy(void* addr) {
    ITT_FUNCTION_STAT();

    __itt_id id = __itt_id_make(addr, 0);
    CTraceEventFormat::SRegularFields rf = GetRegularFields();
    WriteRecord(ERecordType::ObjectDelete, SRecord{rf, *g_pIntelSEAPIDomain, id, __itt_null});
}

inline void SyncState(void* addr, const char* state) {
    ITT_FUNCTION_STAT();

    __itt_id id = __itt_id_make(addr, 0);

    CTraceEventFormat::SRegularFields rf = GetRegularFields();
    WriteRecord(ERecordType::ObjectSnapshot,
                SRecord{rf, *g_pIntelSEAPIDomain, id, __itt_null, nullptr, nullptr, state, strlen(state)});
}

void UNICODE_AGNOSTIC(sync_rename)(void* addr, const char* name) {
    ITT_FUNCTION_STAT();

    SyncState(addr, (std::string("name=") + name).c_str());
}
#ifdef _WIN32
void sync_renameW(void* addr, const wchar_t* name) {
    UNICODE_AGNOSTIC(sync_rename)(addr, W2L(name).c_str());
}
#endif

void sync_prepare(void* addr) {
    ITT_FUNCTION_STAT();

    SyncState(addr, "state=prepare");
}

void sync_cancel(void* addr) {
    ITT_FUNCTION_STAT();

    SyncState(addr, "state=cancel");
}

void sync_acquired(void* addr) {
    ITT_FUNCTION_STAT();
    SyncState(addr, "state=acquired");
}

void sync_releasing(void* addr) {
    ITT_FUNCTION_STAT();
    SyncState(addr, "state=releasing");
}

// region is the same as frame only explicitly named
void region_begin(const __itt_domain* pDomain, __itt_id id, __itt_id parentid, const __itt_string_handle* pName) {
    ITT_FUNCTION_STAT();

    CTraceEventFormat::SRegularFields rf = GetRegularFields();
    WriteRecord(ERecordType::BeginFrame, SRecord{rf, *pDomain, id, parentid, pName});
}

void region_end(const __itt_domain* pDomain, __itt_id id) {
    ITT_FUNCTION_STAT();

    CTraceEventFormat::SRegularFields rf = GetRegularFields();
    WriteRecord(ERecordType::EndFrame, SRecord{rf, *pDomain, id, __itt_null});
}

__itt_clock_domain* clock_domain_create(__itt_get_clock_info_fn fn, void* fn_data) {
    ITT_FUNCTION_STAT();
    CIttLocker lock;
    __itt_domain* pDomain = g_pIntelSEAPIDomain;
    DomainExtra* pDomainExtra = (DomainExtra*)pDomain->extra2;
    __itt_clock_domain** ppClockDomain = &pDomainExtra->pClockDomain;
    while (*ppClockDomain && (*ppClockDomain)->next) {
        ppClockDomain = &(*ppClockDomain)->next;
    }

    __itt_clock_info ci = {};
    uint64_t now1 = CTraceEventFormat::GetRegularFields().nanoseconds;
    fn(&ci, fn_data);
    uint64_t now2 = CTraceEventFormat::GetRegularFields().nanoseconds;

    *ppClockDomain = new __itt_clock_domain{
        ci,
        fn,
        fn_data,
        0,
        new uint64_t((now1 + now2) / 2)  // let's keep current time point in extra2
    };

    return *ppClockDomain;
}

void clock_domain_reset() {
    ITT_FUNCTION_STAT();

    TraverseDomains([](__itt_domain& domain) {
        DomainExtra* pDomainExtra = (DomainExtra*)domain.extra2;
        if (!pDomainExtra)
            return;
        __itt_clock_domain* pClockDomain = pDomainExtra->pClockDomain;
        while (pClockDomain) {
            uint64_t now1 = CTraceEventFormat::GetRegularFields().nanoseconds;
            pClockDomain->fn(&pClockDomain->info, pClockDomain->fn_data);
            uint64_t now2 = CTraceEventFormat::GetRegularFields().nanoseconds;
            *(uint64_t*)pClockDomain->extra2 = (now1 + now2) / 2;
            pClockDomain = pClockDomain->next;
        }
    });
}

void task_begin_ex(const __itt_domain* pDomain,
                   __itt_clock_domain* clock_domain,
                   unsigned long long timestamp,
                   __itt_id taskid,
                   __itt_id parentid,
                   __itt_string_handle* pName) {
    ITT_FUNCTION_STAT();

    SThreadRecord* pThreadRecord = GetThreadRecord();

    CTraceEventFormat::SRegularFields rf = GetRegularFields(clock_domain, timestamp);

    pThreadRecord->pTask = placement_new(STaskDescriptor){pThreadRecord->pTask,  // chaining the previous task inside
                                                          rf,
                                                          pDomain,
                                                          pName,
                                                          taskid,
                                                          parentid};

    for (size_t i = 0; (i < MAX_HANDLERS) && g_handlers[i]; ++i) {
        g_handlers[i]->TaskBegin(*pThreadRecord->pTask, false);
    }
}

void task_end_ex(const __itt_domain* pDomain, __itt_clock_domain* clock_domain, unsigned long long timestamp) {
    ITT_FUNCTION_STAT();

    CTraceEventFormat::SRegularFields rf = GetRegularFields(clock_domain, timestamp);

    SThreadRecord* pThreadRecord = GetThreadRecord();
    if (!pThreadRecord->pTask) {
        VerbosePrint("Uneven begin/end count for domain: %s\n", pDomain->nameA);
        return;
    }
    for (size_t i = 0; (i < MAX_HANDLERS) && g_handlers[i]; ++i) {
        g_handlers[i]->TaskEnd(*pThreadRecord->pTask, rf, false);
    }
    STaskDescriptor* prev = pThreadRecord->pTask->prev;
    placement_free(pThreadRecord->pTask);
    pThreadRecord->pTask = prev;
}

void id_create(const __itt_domain* pDomain, __itt_id id) {
    ITT_FUNCTION_STAT();
    // noting to do here yet
}

void id_destroy(const __itt_domain* pDomain, __itt_id id) {
    ITT_FUNCTION_STAT();
    // noting to do here yet
}

void set_track(__itt_track* track) {
    ITT_FUNCTION_STAT();
    GetThreadRecord()->pTrack = track;
}

int64_t g_lastPseudoThread = -1;
int64_t g_lastPseudoProcess = -1;

__itt_track_group* track_group_create(__itt_string_handle* pName, __itt_track_group_type track_group_type) {
    ITT_FUNCTION_STAT();
    CIttLocker lock;
    __itt_domain* pDomain = g_pIntelSEAPIDomain;
    DomainExtra* pDomainExtra = (DomainExtra*)pDomain->extra2;
    __itt_track_group** ppTrackGroup = &pDomainExtra->pTrackGroup;
    while (*ppTrackGroup && (*ppTrackGroup)->next) {
        if ((*ppTrackGroup)->name == pName)
            return *ppTrackGroup;
        ppTrackGroup = &(*ppTrackGroup)->next;
    }
    if (pName) {
        WriteGroupName(g_lastPseudoProcess, pName->strA);
    }
    // zero name means current process
    return *ppTrackGroup =
               new __itt_track_group{pName, nullptr, track_group_type, int(pName ? g_lastPseudoProcess-- : g_PID)};
}

__itt_track* track_create(__itt_track_group* track_group, __itt_string_handle* name, __itt_track_type track_type) {
    ITT_FUNCTION_STAT();
    CIttLocker locker;

    if (!track_group) {
        track_group = track_group_create(nullptr, __itt_track_group_type_normal);
    }

    __itt_track** ppTrack = &track_group->track;
    while (*ppTrack && (*ppTrack)->next) {
        if ((*ppTrack)->name == name)
            return *ppTrack;
        ppTrack = &(*ppTrack)->next;
    }

    CTraceEventFormat::SRegularFields* pRF =
        new CTraceEventFormat::SRegularFields{int64_t(track_group->extra1), g_lastPseudoThread--};

    for (size_t i = 0; (i < MAX_HANDLERS) && g_handlers[i]; ++i) {
        g_handlers[i]->SetThreadName(*pRF, name->strA);
    }

    return *ppTrack = new __itt_track{name, track_group, track_type, 0, pRF};
}

class COverlapped {
public:
    static COverlapped& Get() {
        SThreadRecord* pThreadRecord = GetThreadRecord();
        if (pThreadRecord->pOverlapped)
            return *pThreadRecord->pOverlapped;
        return *(pThreadRecord->pOverlapped = new COverlapped);
    }

    void Begin(__itt_id taskid,
               const CTraceEventFormat::SRegularFields& rf,
               const __itt_domain* domain,
               __itt_string_handle* name,
               __itt_id parentid) {
        m_map[taskid].reset(placement_new(STaskDescriptor){nullptr,  // chaining the previous task inside
                                                           rf,
                                                           domain,
                                                           name,
                                                           taskid,
                                                           parentid},
                            placement_free<STaskDescriptor>);

        for (size_t i = 0; (i < MAX_HANDLERS) && g_handlers[i]; ++i) {
            g_handlers[i]->TaskBegin(*m_map[taskid], true);
        }
    }

    bool AddArg(const __itt_domain* domain, __itt_id id, __itt_string_handle* key, const char* data, size_t length) {
        TTaskMap::iterator it = m_map.find(id);
        if (m_map.end() == it)
            return false;
        for (size_t i = 0; (i < MAX_HANDLERS) && g_handlers[i]; ++i) {
            g_handlers[i]->AddArg(*m_map[id], key, data, length);
        }
        return true;
    }

    bool AddArg(const __itt_domain* domain, __itt_id id, __itt_string_handle* key, double value) {
        TTaskMap::iterator it = m_map.find(id);
        if (m_map.end() == it)
            return false;
        for (size_t i = 0; (i < MAX_HANDLERS) && g_handlers[i]; ++i) {
            g_handlers[i]->AddArg(*m_map[id], key, value);
        }
        return true;
    }

    void End(__itt_id taskid, const CTraceEventFormat::SRegularFields& rf, const __itt_domain* domain) {
        TTaskMap::iterator it = m_map.find(taskid);
        if (m_map.end() == it)
            return;
        for (size_t i = 0; (i < MAX_HANDLERS) && g_handlers[i]; ++i) {
            g_handlers[i]->TaskEnd(*m_map[taskid], rf, true);
        }
        m_map.erase(it);
    }

    static void FinishAll() {
        TraverseThreadRecords([](SThreadRecord& record) {
            if (record.pOverlapped)
                record.pOverlapped->Finish();
        });
    }

protected:
    void Finish() {
        CTraceEventFormat::SRegularFields rf = CTraceEventFormat::GetRegularFields();
        for (const auto& pair : m_map) {
            for (size_t i = 0; (i < MAX_HANDLERS) && g_handlers[i]; ++i) {
                g_handlers[i]->TaskEnd(*pair.second, rf, true);
            }
        }
        m_map.clear();
    }

    typedef std::map<__itt_id, std::shared_ptr<STaskDescriptor>> TTaskMap;
    TTaskMap m_map;
};

void task_begin_overlapped_ex(const __itt_domain* pDomain,
                              __itt_clock_domain* clock_domain,
                              unsigned long long timestamp,
                              __itt_id taskid,
                              __itt_id parentid,
                              __itt_string_handle* pName) {
    ITT_FUNCTION_STAT();

    COverlapped::Get().Begin(taskid, GetRegularFields(clock_domain, timestamp), pDomain, pName, parentid);
}

void task_begin_overlapped(const __itt_domain* pDomain,
                           __itt_id taskid,
                           __itt_id parentid,
                           __itt_string_handle* pName) {
    ITT_FUNCTION_STAT();

    task_begin_overlapped_ex(pDomain, nullptr, 0, taskid, parentid, pName);
}

void task_end_overlapped_ex(const __itt_domain* pDomain,
                            __itt_clock_domain* clock_domain,
                            unsigned long long timestamp,
                            __itt_id taskid) {
    ITT_FUNCTION_STAT();

    COverlapped::Get().End(taskid, GetRegularFields(clock_domain, timestamp), pDomain);
}

void task_end_overlapped(const __itt_domain* pDomain, __itt_id taskid) {
    ITT_FUNCTION_STAT();

    task_end_overlapped_ex(pDomain, nullptr, 0, taskid);
}

std::map<__itt_id, __itt_string_handle*> g_namedIds;

void SetIdName(const __itt_id& id, const char* data) {
    CIttLocker lock;
    g_namedIds[id] = UNICODE_AGNOSTIC(string_handle_create)(data);
}

template <class... Args>
void MetadataAdd(const __itt_domain* pDomain, __itt_id id, __itt_string_handle* pKey, Args... args) {
    if (id.d1 || id.d2) {
        SThreadRecord* pThreadRecord = GetThreadRecord();
        if (!COverlapped::Get().AddArg(pDomain, id, pKey, args...) && pThreadRecord->pTask &&
            pThreadRecord->pTask->id == id) {
            for (size_t i = 0; (i < MAX_HANDLERS) && g_handlers[i]; ++i) {
                g_handlers[i]->AddArg(*pThreadRecord->pTask, pKey, args...);
            }
        }
    }
}

void UNICODE_AGNOSTIC(metadata_str_add)(const __itt_domain* pDomain,
                                        __itt_id id,
                                        __itt_string_handle* pKey,
                                        const char* data,
                                        size_t length) {
    ITT_FUNCTION_STAT();

    if (id == __itt_null) {
        if (0 == strcmp(pKey->strA, "__sea_cut")) {
            marker(pDomain, id, pKey, __itt_marker_scope_process);
            SetCutName(data);
            return;
        }
        if (0 == strcmp(pKey->strA, "__sea_set_folder")) {
            SetFolder(data);
            return;
        }
        if (0 == strcmp(pKey->strA, "__sea_set_ring")) {
            SetRing(1000000000ull * atoi(data));
            return;
        }
        if (0 == strcmp(pKey->strA, "__sea_ftrace_sync")) {
#ifdef __linux__
            WriteFTraceTimeSyncMarkers();
#endif
            return;
        }
    }
    if (!length)
        length = data ? strlen(data) : 0;
    if (!pKey)
        SetIdName(id, data);
    else
        MetadataAdd(pDomain, id, pKey, data, length);
}

#ifdef _WIN32
void metadata_str_addW(const __itt_domain* pDomain,
                       __itt_id id,
                       __itt_string_handle* pKey,
                       const wchar_t* data,
                       size_t length) {
    UNICODE_AGNOSTIC(metadata_str_add)(pDomain, id, pKey, W2L(data).c_str(), length);
}
#endif

void metadata_add(const __itt_domain* pDomain,
                  __itt_id id,
                  __itt_string_handle* pKey,
                  __itt_metadata_type type,
                  size_t count,
                  void* data) {
    ITT_FUNCTION_STAT();

    if (id.d1 || id.d2) {
        if (data) {
            if (__itt_metadata_unknown != type) {
                double res = g_MetatypeFormatConverter[type](data);
                MetadataAdd(pDomain, id, pKey, res);
            } else {
                if (count)
                    MetadataAdd(pDomain, id, pKey, (const char*)data, count);  // raw data with size
                else
                    MetadataAdd(pDomain, id, pKey, (double)(uint64_t)data);  // just pointer, convert it to number
            }
        }
    } else {
        if (__itt_metadata_unknown == type)
            return;
        Counter(pDomain, pKey, g_MetatypeFormatConverter[type](data));
    }
}

const char* api_version(void) {
    ITT_FUNCTION_STAT();
    return "IntelSEAPI";
}

void frame_begin_v3(const __itt_domain* pDomain, __itt_id* id) {
    ITT_FUNCTION_STAT();

    CTraceEventFormat::SRegularFields rf = GetRegularFields();
    WriteRecord(ERecordType::BeginFrame, SRecord{rf, *pDomain, id ? *id : __itt_null, __itt_null});
}

void frame_end_v3(const __itt_domain* pDomain, __itt_id* id) {
    ITT_FUNCTION_STAT();

    CTraceEventFormat::SRegularFields rf = GetRegularFields();
    WriteRecord(ERecordType::EndFrame, SRecord{rf, *pDomain, id ? *id : __itt_null, __itt_null});
}

__itt_frame_t* UNICODE_AGNOSTIC(frame_create)(const char* domain) {
    ITT_FUNCTION_STAT();
    return new __itt_frame_t{UNICODE_AGNOSTIC(domain_create)(domain), __itt_id_make(const_cast<char*>(domain), 0)};
}

#ifdef _WIN32
__itt_frame_t* frame_createW(const wchar_t* domain) {
    return UNICODE_AGNOSTIC(frame_create)(W2L(domain).c_str());
}
#endif

void frame_begin(__itt_frame_t* frame) {
    ITT_FUNCTION_STAT();
    frame_begin_v3(frame->pDomain, &frame->id);
}

void frame_end(__itt_frame_t* frame) {
    ITT_FUNCTION_STAT();
    frame_end_v3(frame->pDomain, &frame->id);
}

void frame_submit_v3(const __itt_domain* pDomain, __itt_id* pId, __itt_timestamp begin, __itt_timestamp end) {
    ITT_FUNCTION_STAT();

    CTraceEventFormat::SRegularFields rf = GetRegularFields();
    if (__itt_timestamp_none == end)
        end = rf.nanoseconds;
    const __itt_string_handle* pName = nullptr;
    if (pId) {
        if (pId->d3) {
            pName = reinterpret_cast<__itt_string_handle*>(pId->d3);
        } else {
            CIttLocker lock;
            auto it = g_namedIds.find(*pId);
            if (g_namedIds.end() != it) {
                pName = it->second;
                pId->d3 = (unsigned long long)pName;
            }
        }
    }
    rf.nanoseconds = begin;
    WriteRecord(ERecordType::BeginFrame, SRecord{rf, *pDomain, pId ? *pId : __itt_null, __itt_null, pName});
    rf.nanoseconds = end;
    WriteRecord(ERecordType::EndFrame, SRecord{rf, *pDomain, pId ? *pId : __itt_null, __itt_null});
}

__itt_timestamp get_timestamp() {
    ITT_FUNCTION_STAT();
    return GetRegularFields().nanoseconds;
}

void Pause() {
    static __itt_global* pGlobal = GetITTGlobal();
    while (pGlobal) {
        pGlobal->state = __itt_collection_init_fail;
        ___itt_domain* pDomain = pGlobal->domain_list;
        while (pDomain) {
            pDomain->flags =
                0;  // this flag is analyzed by static part of ITT to decide where to call dynamic part or not
            pDomain = pDomain->next;
        }
        pGlobal = pGlobal->next;
    }
}

void pause() {
    ITT_FUNCTION_STAT();
    static __itt_string_handle* pPause = UNICODE_AGNOSTIC(string_handle_create)("PAUSE");
    static __itt_global* pGlobal = GetITTGlobal();
    static __itt_id id = __itt_id_make(pGlobal, 0);
    region_begin(pGlobal->domain_list, id, __itt_null, pPause);
    Pause();
}

void Resume() {
    static __itt_global* pGlobal = GetITTGlobal();

    while (pGlobal) {
        ___itt_domain* pDomain = pGlobal->domain_list;
        while (pDomain) {
            pDomain->flags =
                1;  // this flag is analyzed by static part of ITT to decide where to call dynamic part or not
            pDomain = pDomain->next;
        }
        pGlobal->state = __itt_collection_uninitialized;
        pGlobal = pGlobal->next;
    }
}

void resume() {
    ITT_FUNCTION_STAT();
    static __itt_global* pGlobal = GetITTGlobal();
    static __itt_id id = __itt_id_make(pGlobal, 0);
    region_end(pGlobal->domain_list, id);
    Resume();
}

using TRelations = __itt_string_handle * [__itt_relation_is_predecessor_to + 1];
// it's not static member of function to avoid racing
TRelations g_relations = {};  // will be filled in InitSEA

void relation_add_ex(const __itt_domain* pDomain,
                     __itt_clock_domain* clock_domain,
                     unsigned long long timestamp,
                     __itt_id head,
                     __itt_relation relation,
                     __itt_id tail) {
    ITT_FUNCTION_STAT();
    CTraceEventFormat::SRegularFields rf = GetRegularFields(clock_domain, timestamp);

    for (size_t i = 0; (i < MAX_HANDLERS) && g_handlers[i]; ++i) {
        g_handlers[i]->AddRelation(rf, pDomain, head, g_relations[relation], tail);
    }
}

void relation_add_to_current(const __itt_domain* pDomain, __itt_relation relation, __itt_id tail) {
    ITT_FUNCTION_STAT();
    relation_add_ex(pDomain, nullptr, 0, __itt_null, relation, tail);
}

void relation_add(const __itt_domain* pDomain, __itt_id head, __itt_relation relation, __itt_id tail) {
    ITT_FUNCTION_STAT();
    relation_add_ex(pDomain, nullptr, 0, head, relation, tail);
}

void relation_add_to_current_ex(const __itt_domain* pDomain,
                                __itt_clock_domain* clock_domain,
                                unsigned long long timestamp,
                                __itt_relation relation,
                                __itt_id tail) {
    ITT_FUNCTION_STAT();
    relation_add_ex(pDomain, clock_domain, timestamp, __itt_null, relation, tail);
}

struct SHeapFunction {
    __itt_domain* pDomain;
    std::string name;
    ___itt_string_handle* pName;
};

__itt_heap_function ITTAPI UNICODE_AGNOSTIC(heap_function_create)(const char* name, const char* domain) {
    ITT_FUNCTION_STAT();
    std::string counter_name = std::string(name) + ":ALL(bytes)";
    return new SHeapFunction{UNICODE_AGNOSTIC(domain_create)(domain),
                             name,
                             UNICODE_AGNOSTIC(string_handle_create)(counter_name.c_str())};
}

#ifdef _WIN32
__itt_heap_function ITTAPI heap_function_createW(const wchar_t* name, const wchar_t* domain) {
    return UNICODE_AGNOSTIC(heap_function_create)(W2L(name).c_str(), W2L(domain).c_str());
}
#endif

class CMemoryTracker {
protected:
    TCritSec m_cs;

    typedef std::pair<const __itt_domain*, const void* /*task name or function pointer*/> TDomainString;

    struct SNode {
        struct SMemory {
            int32_t current_amount = 0;
            int32_t max_amount = 0;
        };
        std::map<size_t, SMemory> memory;
        std::map<TDomainString, SNode> chilren;
    };
    SNode m_tree;

    std::map<const void*, std::pair<size_t, SNode*>> m_size_map;
    typedef std::pair<__itt_string_handle*, size_t /*count*/> TBlockData;
    std::map<size_t /*block size*/, TBlockData> m_counter_map;
    bool m_bInitialized = false;

public:
    CMemoryTracker() : m_bInitialized(true) {}
    void Alloc(SHeapFunction* pHeapFunction, const void* addr, size_t size) {
        static bool bMemCount = !!(GetFeatureSet() & sfMemCounters);

        if (!m_bInitialized)
            return;

        for (size_t i = 0; (i < MAX_HANDLERS) && g_handlers[i]; ++i) {
            g_handlers[i]->Alloc(GetRegularFields(),
                                 addr,
                                 size,
                                 pHeapFunction->pDomain->nameA,
                                 pHeapFunction->name.c_str());
        }

        SNode* pNode = UpdateAllocation(size, +1, nullptr);
        TBlockData block;
        {
            std::lock_guard<TCritSec> lock(m_cs);
            m_size_map[addr] = std::make_pair(size, pNode);
            if (bMemCount) {
                auto it = m_counter_map.find(size);
                if (m_counter_map.end() == it) {
                    std::string name = pHeapFunction->name + std::string(":size<") + std::to_string(size) + ">(count)";
                    __itt_string_handle* pName = UNICODE_AGNOSTIC(string_handle_create)(name.c_str());
                    it = m_counter_map.insert(m_counter_map.end(),
                                              std::make_pair(size, std::make_pair(pName, size_t(1))));
                } else {
                    ++it->second.second;
                }
                block = it->second;
            }
        }
        if (bMemCount) {
            Counter(pHeapFunction->pDomain, block.first, double(block.second));  // report current count for this size
        }
    }

    SNode* UpdateAllocation(size_t size, int32_t delta, SNode* pNode) {
        static bool bMemStat = (GetFeatureSet() & sfMemStat) && InitMemStat();
        if (!bMemStat)
            return nullptr;
        SThreadRecord* pThreadRecord = GetThreadRecord();
        STaskDescriptor* pTask = pThreadRecord->pTask;
        std::stack<TDomainString> stack;
        if (!pNode) {
            for (; pTask; pTask = pTask->prev) {
                stack.push(TDomainString(pTask->pDomain, pTask->pName ? pTask->pName : pTask->fn));
            }
        }
        std::lock_guard<TCritSec> lock(m_cs);
        if (!pNode) {
            pNode = &m_tree;
            while (!stack.empty()) {
                pNode = &m_tree.chilren[stack.top()];
                stack.pop();
            }
        }
        SNode::SMemory& mem = pNode->memory[size];
        mem.current_amount += delta;
        if (mem.current_amount > mem.max_amount)
            mem.max_amount = mem.current_amount;
        return pNode;
    }

    void Free(SHeapFunction* pHeapFunction, const void* addr) {
        static bool bMemCount = !!(GetFeatureSet() & sfMemCounters);
        size_t size = 0;
        if (m_bInitialized) {
            std::lock_guard<TCritSec> lock(m_cs);

            const auto& pair = m_size_map[addr];
            size = pair.first;
            SNode* pNode = pair.second;
            m_size_map.erase(addr);
            if (bMemCount) {
                auto it = m_counter_map.find(size);
                if (m_counter_map.end() == it)
                    return;  // how come?
                else
                    --it->second.second;
                Counter(pHeapFunction->pDomain, it->second.first, double(it->second.second));
            }
            if (pNode)  // if we missed allocation, we don't care about freeing
                UpdateAllocation(size, -1, pNode);
        }
        for (size_t i = 0; (i < MAX_HANDLERS) && g_handlers[i]; ++i) {
            g_handlers[i]->Free(GetRegularFields(),
                                addr,
                                size,
                                pHeapFunction->pDomain->nameA,
                                pHeapFunction->name.c_str());
        }
    }

    void SaveMemoryStatistics() {
        if (!(GetFeatureSet() & sfMemStat))
            return;
        std::lock_guard<TCritSec> lock(m_cs);
        WriteNode(m_tree);
    }

    template <class T>
    void WriteMem(T value) {
        WriteMemStat(&value, sizeof(T));
    }

    void WriteNode(const SNode& node) {
        WriteMem((uint32_t)node.memory.size());
        for (const auto& pair : node.memory) {
            WriteMem((uint32_t)pair.first);              // size
            WriteMem(pair.second.current_amount);        // SNode::SMemory
            WriteMem((uint32_t)pair.second.max_amount);  // SNode::SMemory
        }
        WriteMem((uint32_t)node.chilren.size());
        for (const auto& pair : node.chilren) {
            const TDomainString& domain_string = pair.first;
            WriteMem((const void*)domain_string.first);   // domain
            WriteMem((const void*)domain_string.second);  // string
            WriteNode(pair.second);
        }
    }

    ~CMemoryTracker() {
        m_bInitialized = false;
        SaveMemoryStatistics();
    }
} g_oMemoryTracker;

void heap_allocate_begin(__itt_heap_function h, size_t size, int initialized) {
    ITT_FUNCTION_STAT();
}

void heap_allocate_end(__itt_heap_function h, void** addr, size_t size, int) {
    ITT_FUNCTION_STAT();
    g_oMemoryTracker.Alloc(reinterpret_cast<SHeapFunction*>(h), *addr, size);
}

void heap_free_begin(__itt_heap_function h, void* addr) {
    ITT_FUNCTION_STAT();
    g_oMemoryTracker.Free(reinterpret_cast<SHeapFunction*>(h), addr);
}

void heap_free_end(__itt_heap_function h, void* addr) {
    ITT_FUNCTION_STAT();
}

__itt_domain* get_events_domain() {
    static __itt_domain* s_pEvents = UNICODE_AGNOSTIC(domain_create)("sea_events");
    return s_pEvents;
}

__itt_event UNICODE_AGNOSTIC(event_create)(const char* name, int namelen) {
    ITT_FUNCTION_STAT();
    __itt_domain* pEvents = get_events_domain();
    __itt_string_handle* pStr = UNICODE_AGNOSTIC(string_handle_create)(name);
    return __itt_event(intptr_t(pStr) - intptr_t(pEvents));
}

int event_start(__itt_event event) {
    ITT_FUNCTION_STAT();
    __itt_domain* pEvents = get_events_domain();
    __itt_string_handle* pStr = reinterpret_cast<__itt_string_handle*>(intptr_t(pEvents) + event);
    task_begin_overlapped(pEvents, __itt_id_make(pEvents, (unsigned long long)pStr), __itt_null, pStr);
    return event;
}

int event_end(__itt_event event) {
    ITT_FUNCTION_STAT();
    __itt_domain* pEvents = get_events_domain();
    __itt_string_handle* pStr = reinterpret_cast<__itt_string_handle*>(intptr_t(pEvents) + event);
    task_end_overlapped(pEvents, __itt_id_make(pEvents, (unsigned long long)pStr));
    return event;
}

#ifdef _WIN32
__itt_event ITTAPI event_createW(const wchar_t* name, int namelen) {
    return UNICODE_AGNOSTIC(event_create)(W2L(name).c_str(), namelen);
}
#endif

#ifdef _WIN32
#    define WIN(something) something
#else
#    define WIN(nothing)
#endif

#define _AW(macro, name) macro(UNICODE_AGNOSTIC(name)) WIN(macro(ITT_JOIN(name, W)))

#define ORIGINAL_FUNCTIONS()                                   \
    ITT_STUB_IMPL_ORIG(UNICODE_AGNOSTIC(domain_create))        \
    WIN(ITT_STUB_IMPL_ORIG(domain_createW))                    \
    ITT_STUB_IMPL_ORIG(UNICODE_AGNOSTIC(string_handle_create)) \
    WIN(ITT_STUB_IMPL_ORIG(string_handle_createW))

#define API_MAP()                                      \
    _AW(ITT_STUB_IMPL, thread_set_name)                \
    ITT_STUB_IMPL(task_begin)                          \
    ITT_STUB_IMPL(task_begin_fn)                       \
    ITT_STUB_IMPL(task_end)                            \
    _AW(ITT_STUB_IMPL, metadata_str_add)               \
    ITT_STUB_IMPL(marker)                              \
    ITT_STUB_IMPL(marker_ex)                           \
    ITT_STUB_IMPL(counter_inc_delta_v3)                \
    _AW(ITT_STUB_IMPL, counter_create)                 \
    _AW(ITT_STUB_IMPL, counter_create_typed)           \
    ITT_STUB_IMPL(counter_set_value)                   \
    ITT_STUB_IMPL(counter_set_value_ex)                \
    ITT_STUB_IMPL(clock_domain_create)                 \
    ITT_STUB_IMPL(clock_domain_reset)                  \
    ITT_STUB_IMPL(task_begin_ex)                       \
    ITT_STUB_IMPL(task_end_ex)                         \
    ITT_STUB_IMPL(id_create)                           \
    ITT_STUB_IMPL(set_track)                           \
    ITT_STUB_IMPL(track_create)                        \
    ITT_STUB_IMPL(track_group_create)                  \
    ITT_STUB_IMPL(task_begin_overlapped)               \
    ITT_STUB_IMPL(task_begin_overlapped_ex)            \
    ITT_STUB_IMPL(task_end_overlapped)                 \
    ITT_STUB_IMPL(task_end_overlapped_ex)              \
    ITT_STUB_IMPL(id_destroy)                          \
    ITT_STUB_IMPL(api_version)                         \
    ITT_STUB_IMPL(frame_begin_v3)                      \
    ITT_STUB_IMPL(frame_end_v3)                        \
    ITT_STUB_IMPL(frame_submit_v3)                     \
    _AW(ITT_STUB_IMPL, frame_create)                   \
    ITT_STUB_IMPL(frame_begin)                         \
    ITT_STUB_IMPL(frame_end)                           \
    ITT_STUB_IMPL(region_begin)                        \
    ITT_STUB_IMPL(region_end)                          \
    ITT_STUB_IMPL(pause)                               \
    ITT_STUB_IMPL(resume)                              \
    ITT_STUB_IMPL(get_timestamp)                       \
    ITT_STUB_IMPL(metadata_add)                        \
    _AW(ITT_STUB_IMPL, sync_create)                    \
    ITT_STUB_IMPL(sync_destroy)                        \
    ITT_STUB_IMPL(sync_acquired)                       \
    ITT_STUB_IMPL(sync_releasing)                      \
    _AW(ITT_STUB_IMPL, sync_rename)                    \
    ITT_STUB_IMPL(sync_prepare)                        \
    ITT_STUB_IMPL(sync_cancel)                         \
    ITT_STUB_IMPL(relation_add_to_current)             \
    ITT_STUB_IMPL(relation_add)                        \
    ITT_STUB_IMPL(relation_add_to_current_ex)          \
    ITT_STUB_IMPL(relation_add_ex)                     \
    _AW(ITT_STUB_IMPL, heap_function_create)           \
    ITT_STUB_IMPL(heap_allocate_begin)                 \
    ITT_STUB_IMPL(heap_allocate_end)                   \
    ITT_STUB_IMPL(heap_free_begin)                     \
    ITT_STUB_IMPL(heap_free_end)                       \
    _AW(ITT_STUB_IMPL, event_create)                   \
    WIN(_AW(ITT_STUB_IMPL, event_create))              \
    ITT_STUB_IMPL(event_start)                         \
    ITT_STUB_IMPL(event_end)                           \
    ORIGINAL_FUNCTIONS()                               \
    ITT_STUB_NO_IMPL(thread_ignore)                    \
    _AW(ITT_STUB_NO_IMPL, thr_name_set)                \
    ITT_STUB_NO_IMPL(thr_ignore)                       \
    ITT_STUB_NO_IMPL(counter_inc_delta)                \
    ITT_STUB_NO_IMPL(enable_attach)                    \
    ITT_STUB_NO_IMPL(suppress_push)                    \
    ITT_STUB_NO_IMPL(suppress_pop)                     \
    ITT_STUB_NO_IMPL(suppress_mark_range)              \
    ITT_STUB_NO_IMPL(suppress_clear_range)             \
    ITT_STUB_NO_IMPL(model_site_beginA)                \
    WIN(ITT_STUB_NO_IMPL(model_site_beginW))           \
    ITT_STUB_NO_IMPL(model_site_beginAL)               \
    ITT_STUB_NO_IMPL(model_site_end)                   \
    _AW(ITT_STUB_NO_IMPL, model_task_begin)            \
    ITT_STUB_NO_IMPL(model_task_end)                   \
    ITT_STUB_NO_IMPL(model_lock_acquire)               \
    ITT_STUB_NO_IMPL(model_lock_release)               \
    ITT_STUB_NO_IMPL(model_record_allocation)          \
    ITT_STUB_NO_IMPL(model_record_deallocation)        \
    ITT_STUB_NO_IMPL(model_induction_uses)             \
    ITT_STUB_NO_IMPL(model_reduction_uses)             \
    ITT_STUB_NO_IMPL(model_observe_uses)               \
    ITT_STUB_NO_IMPL(model_clear_uses)                 \
    ITT_STUB_NO_IMPL(model_site_begin)                 \
    ITT_STUB_NO_IMPL(model_site_beginA)                \
    WIN(ITT_STUB_NO_IMPL(model_site_beginW))           \
    ITT_STUB_NO_IMPL(model_site_beginAL)               \
    ITT_STUB_NO_IMPL(model_task_begin)                 \
    ITT_STUB_NO_IMPL(model_task_beginA)                \
    WIN(ITT_STUB_NO_IMPL(model_task_beginW))           \
    ITT_STUB_NO_IMPL(model_task_beginAL)               \
    ITT_STUB_NO_IMPL(model_iteration_taskA)            \
    WIN(ITT_STUB_NO_IMPL(model_iteration_taskW))       \
    ITT_STUB_NO_IMPL(model_iteration_taskAL)           \
    ITT_STUB_NO_IMPL(model_site_end_2)                 \
    ITT_STUB_NO_IMPL(model_task_end_2)                 \
    ITT_STUB_NO_IMPL(model_lock_acquire_2)             \
    ITT_STUB_NO_IMPL(model_lock_release_2)             \
    ITT_STUB_NO_IMPL(model_aggregate_task)             \
    ITT_STUB_NO_IMPL(model_disable_push)               \
    ITT_STUB_NO_IMPL(model_disable_pop)                \
    ITT_STUB_NO_IMPL(heap_reallocate_begin)            \
    ITT_STUB_NO_IMPL(heap_reallocate_end)              \
    ITT_STUB_NO_IMPL(heap_internal_access_begin)       \
    ITT_STUB_NO_IMPL(heap_internal_access_end)         \
    ITT_STUB_NO_IMPL(heap_record_memory_growth_begin)  \
    ITT_STUB_NO_IMPL(heap_record_memory_growth_end)    \
    ITT_STUB_NO_IMPL(heap_reset_detection)             \
    ITT_STUB_NO_IMPL(heap_record)                      \
    ITT_STUB_NO_IMPL(task_group)                       \
    ITT_STUB_NO_IMPL(counter_inc_v3)                   \
    _AW(ITT_STUB_NO_IMPL, sync_set_name)               \
    _AW(ITT_STUB_NO_IMPL, notify_sync_name)            \
    ITT_STUB_NO_IMPL(notify_sync_prepare)              \
    ITT_STUB_NO_IMPL(notify_sync_cancel)               \
    ITT_STUB_NO_IMPL(notify_sync_acquired)             \
    ITT_STUB_NO_IMPL(notify_sync_releasing)            \
    ITT_STUB_NO_IMPL(memory_read)                      \
    ITT_STUB_NO_IMPL(memory_write)                     \
    ITT_STUB_NO_IMPL(memory_update)                    \
    ITT_STUB_NO_IMPL(state_get)                        \
    ITT_STUB_NO_IMPL(state_set)                        \
    ITT_STUB_NO_IMPL(obj_mode_set)                     \
    ITT_STUB_NO_IMPL(thr_mode_set)                     \
    ITT_STUB_NO_IMPL(counter_destroy)                  \
    ITT_STUB_NO_IMPL(counter_inc)                      \
    ITT_STUB_NO_IMPL(counter_inc_v3)                   \
    _AW(ITT_STUB_NO_IMPL, mark_create)                 \
    _AW(ITT_STUB_NO_IMPL, mark)                        \
    ITT_STUB_NO_IMPL(mark_off)                         \
    _AW(ITT_STUB_NO_IMPL, mark_global)                 \
    ITT_STUB_NO_IMPL(mark_global_off)                  \
    ITT_STUB_NO_IMPL(stack_caller_create)              \
    ITT_STUB_NO_IMPL(stack_caller_destroy)             \
    ITT_STUB_NO_IMPL(stack_callee_enter)               \
    ITT_STUB_NO_IMPL(stack_callee_leave)               \
    ITT_STUB_NO_IMPL(id_create_ex)                     \
    ITT_STUB_NO_IMPL(id_destroy_ex)                    \
    ITT_STUB_NO_IMPL(task_begin_fn_ex)                 \
    ITT_STUB_NO_IMPL(metadata_add_with_scope)          \
    _AW(ITT_STUB_NO_IMPL, metadata_str_add_with_scope) \
    _AW(ITT_STUB_NO_IMPL, av_save)

void FillApiList(__itt_api_info* api_list_ptr) {
#define ITT_STUB_IMPL(fn)                                             \
    if (0 == strcmp("__itt_" ITT_TO_STR(fn), api_list_ptr[i].name)) { \
        *api_list_ptr[i].func_ptr = (void*)sea::fn;                   \
        continue;                                                     \
    }
#define ITT_STUB_IMPL_ORIG(name) ITT_STUB_IMPL(name)
#ifdef _DEBUG  // dangerous stub that doesn't return anything (even when expected) but records the function call for
               // statistics sake
#    define ITT_STUB_NO_IMPL(fn)                                                              \
        if (0 == strcmp("__itt_" ITT_TO_STR(fn), api_list_ptr[i].name)) {                     \
            struct local {                                                                    \
                static void stub(...) { CIttFnStat oIttFnStat("NO IMPL:\t" ITT_TO_STR(fn)); } \
            };                                                                                \
            *api_list_ptr[i].func_ptr = reinterpret_cast<void*>(local::stub);                 \
            continue;                                                                         \
        }
#else
#    define ITT_STUB_NO_IMPL(fn)
#endif

    for (int i = 0; (api_list_ptr[i].name != NULL) && (*api_list_ptr[i].name != 0); ++i) {
        API_MAP();  // continue is called inside when function is found
        VerbosePrint("Not bound: %s\n", api_list_ptr[i].name);
    }
#undef ITT_STUB_IMPL
#undef ITT_STUB_IMPL_ORIG
#undef ITT_STUB_NO_IMPL
}

uint64_t GetFeatureSet() {
    static std::string env = get_environ_value("INTEL_SEA_FEATURES");
    static std::string save = GetSavePath();

    static uint64_t features = (std::string::npos != env.find("mfp") ? sfMetricsFrameworkPublisher : 0) |
                               (std::string::npos != env.find("mfc") ? sfMetricsFrameworkConsumer : 0) |
                               (save.size() ? sfSEA : 0) | (std::string::npos != env.find("stack") ? sfStack : 0) |
                               (std::string::npos != env.find("vscv") ? sfConcurrencyVisualizer : 0) |
                               (std::string::npos != env.find("rmtr") ? sfRemotery : 0) |
                               (std::string::npos != env.find("brflr") ? sfBrofiler : 0) |
                               (std::string::npos != env.find("memstat") ? sfMemStat : 0) |
                               (std::string::npos != env.find("memcount") ? sfMemCounters : 0) |
                               (std::string::npos != env.find("rad") ? sfRadTelemetry : 0);
    return features;
}

void TraverseDomains(const std::function<void(___itt_domain&)>& callback) {
    __itt_global* pGlobal = GetITTGlobal();
    for (___itt_domain* pDomain = pGlobal->domain_list; pDomain; pDomain = pDomain->next) {
        callback(*pDomain);
    }
}

void TraverseThreadRecords(const std::function<void(SThreadRecord&)>& callback) {
    TraverseDomains([&](___itt_domain& domain) {
        if (DomainExtra* pDomainExtra = reinterpret_cast<DomainExtra*>(domain.extra2)) {
            for (SThreadRecord* pThreadRecord = pDomainExtra->pThreadRecords; pThreadRecord;
                 pThreadRecord = pThreadRecord->pNext)
                callback(*pThreadRecord);
        }
    });
}

void SetCutName(const std::string& name) {
    CIttLocker lock;
    g_spCutName = std::make_shared<std::string>(Escape4Path(name));
    TraverseThreadRecords([](SThreadRecord& record) {
        record.nSpeedupCounter =
            (std::numeric_limits<int>::max)();  // changing number is safer than changing pointer to last recorder
    });
}

// in global scope variables are initialized from main thread
// that's the simplest way to get tid of Main Thread
CTraceEventFormat::SRegularFields g_rfMainThread = CTraceEventFormat::GetRegularFields();

void SetFolder(const std::string& path) {
    CIttLocker lock;

    std::string new_path =
        path.size() ? (path + "-" + std::to_string(CTraceEventFormat::GetRegularFields().pid) + "/") : "";

    if (g_savepath == new_path)
        return;

    // To move into a new folder we must make sure next things:
    // 1. per thread files are closed and reopened with new folder
    // 2. strings are reported to new folder
    // 3. domain paths are updated, so that any newly created files would be in right place
    // 4. modules are reported to new folder
    // 5. write process info to the new trace

    g_savepath = new_path;

    for (__itt_global* pGlobal = GetITTGlobal(); pGlobal; pGlobal = pGlobal->next) {
        ReportModule(pGlobal);  // 4. we move to new folder and need to notify modules there

        for (___itt_domain* pDomain = pGlobal->domain_list; pDomain; pDomain = pDomain->next) {
            DomainExtra* pDomainExtra = reinterpret_cast<DomainExtra*>(pDomain->extra2);
            if (pDomainExtra) {
                pDomainExtra->strDomainPath =
                    g_savepath.size() ? GetDir(g_savepath, Escape4Path(pDomain->nameA)) : "";  // 3.
                pDomainExtra->bHasDomainPath = !pDomainExtra->strDomainPath.empty();
                for (SThreadRecord* pThreadRecord = pDomainExtra->pThreadRecords; pThreadRecord;
                     pThreadRecord = pThreadRecord->pNext) {
                    if (g_savepath.size()) {
                        pThreadRecord->bRemoveFiles =
                            true;  // 1. on next attempt to get a file it will recreate all files with new paths
                    } else {
                        pThreadRecord->files.clear();
                    }
                }
            }
        }

        if (g_savepath.size()) {
            for (___itt_string_handle* pString = pGlobal->string_list; pString; pString = pString->next)
                sea::ReportString(const_cast<__itt_string_handle*>(
                    pString));  // 2. making string to be reported again - into the new folder
        }
    }

    if (g_savepath.size())
        GetSEARecorder().Init(g_rfMainThread);  // 5.

    if (g_savepath.size())
        g_features |= sfSEA;
    else
        g_features &= ~sfSEA;
}

void SetRing(uint64_t nanoseconds) {
    if (g_nRingBuffer == nanoseconds)
        return;
    g_nRingBuffer = nanoseconds;
    TraverseThreadRecords([](SThreadRecord& record) {
        record.bRemoveFiles = true;
    });
}

#ifdef __linux__
bool WriteFTraceTimeSyncMarkers() {
    int fd = open("/sys/kernel/debug/tracing/trace_marker", O_WRONLY);
    if (-1 == fd) {
        VerbosePrint("Warning: failed to access /sys/kernel/debug/tracing/trace_marker\n");
        return false;
    }
    for (size_t i = 0; i < 5; ++i) {
        char buff[100] = {};
        int size = snprintf(buff,
                            sizeof(buff),
                            "IntelSEAPI_Time_Sync: %llu\n",
                            (long long unsigned int)CTraceEventFormat::GetTimeNS());
        int res = write(fd, buff, (unsigned int)size);
        if (-1 == res)
            return false;
    }
    close(fd);
    return true;
}
#endif

#ifdef __APPLE__
bool WriteKTraceTimeSyncMarkers() {
    for (size_t i = 0; i < 5; ++i) {
        kdebug_signpost(APPSDBG_CODE(DBG_MACH_CHUD, 0x15EA),
                        CTraceEventFormat::GetTimeNS(),
                        0x15EA15EA,
                        0x15EA15EA,
                        0x15EA15EA);
        syscall(SYS_kdebug_trace,
                APPSDBG_CODE(DBG_MACH_CHUD, 0x15EA) | DBG_FUNC_NONE,
                CTraceEventFormat::GetTimeNS(),
                0x15EA15EA,
                0x15EA15EA,
                0x15EA15EA);
    }
    return true;
}
#endif

void InitSEA() {
    for (size_t i = 0; (i < MAX_HANDLERS) && g_handlers[i]; ++i) {
        g_handlers[i]->Init(g_rfMainThread);
    }
#ifdef __linux__
    WriteFTraceTimeSyncMarkers();
#endif

    const char* relations[] = {
        nullptr,
        ("dependent_on"),    /**< "A is dependent on B" means that A cannot start until B completes */
        ("sibling_of"),      /**< "A is sibling of B" means that A and B were created as a group */
        ("parent_of"),       /**< "A is parent of B" means that A created B */
        ("continuation_of"), /**< "A is continuation of B" means that A assumes the dependencies of B */
        ("child_of"),        /**< "A is child of B" means that A was created by B (inverse of is_parent_of) */
        ("continued_by"),    /**< "A is continued by B" means that B assumes the dependencies of A (inverse of
                                is_continuation_of) */
        ("predecessor_to")   /**< "A is predecessor to B" means that B cannot start until A completes (inverse of
                                is_dependent_on) */
    };

    size_t i = 0;
    for (auto ptr : relations)
        g_relations[i++] = ptr ? UNICODE_AGNOSTIC(string_handle_create)(ptr) : nullptr;

    GetSEARecorder().Init(g_rfMainThread);

#ifdef _WIN32  // adding information about process explicitly
    ReportModule(GetModuleHandle(NULL));
#else
    // XXX ReportModule(dlopen(NULL, RTLD_LAZY));
#endif
#if defined(_DEBUG) && defined(STANDARD_SOURCES) && 0
    void Test();
    Test();
#endif
}

void FinitaLaComedia() {
    COverlapped::FinishAll();

    for (size_t i = 0; (i < MAX_HANDLERS) && g_handlers[i]; ++i) {
        delete g_handlers[i];
        g_handlers[i] = nullptr;
    }

    {
        CIttLocker locker;
        if (sea::IsVerboseMode()) {
            VerbosePrint("Call statistics:\n");
            const auto& map = CIttFnStat::GetStats();
            for (const auto& pair : map) {
                VerbosePrint("%d\t%s\n", (int)pair.second, pair.first.c_str());
            }
        }

        TraverseThreadRecords([](SThreadRecord& tr) {
            tr.files.clear();
        });
    }
#ifdef __linux__
    WriteFTraceTimeSyncMarkers();
#endif

    g_oDomainFilter.Finish();
}
}  // namespace sea

extern "C" {
SEA_EXPORT void* itt_create_domain(const char* str) {
    return UNICODE_AGNOSTIC(__itt_domain_create)(str);
}
SEA_EXPORT void* itt_create_string(const char* str) {
    return UNICODE_AGNOSTIC(__itt_string_handle_create)(str);
}
SEA_EXPORT void itt_marker(void* domain, uint64_t id, void* name, int scope, uint64_t timestamp) {
    __itt_marker_ex(reinterpret_cast<__itt_domain*>(domain),
                    nullptr,  // zero clock domain means that given time is already a correct timestamp
                    timestamp,
                    id ? __itt_id_make(domain, id) : __itt_null,
                    reinterpret_cast<__itt_string_handle*>(name),
                    (__itt_scope)scope);
}

SEA_EXPORT void itt_task_begin(void* domain, uint64_t id, uint64_t parent, void* name, uint64_t timestamp) {
    __itt_task_begin_ex(reinterpret_cast<__itt_domain*>(domain),
                        nullptr,
                        timestamp,
                        id ? __itt_id_make(domain, id) : __itt_null,
                        parent ? __itt_id_make(domain, parent) : __itt_null,
                        reinterpret_cast<__itt_string_handle*>(name));
}

SEA_EXPORT void itt_task_begin_overlapped(void* domain, uint64_t id, uint64_t parent, void* name, uint64_t timestamp) {
    __itt_task_begin_overlapped_ex(reinterpret_cast<__itt_domain*>(domain),
                                   nullptr,
                                   timestamp,
                                   __itt_id_make(domain, id),
                                   parent ? __itt_id_make(domain, parent) : __itt_null,
                                   reinterpret_cast<__itt_string_handle*>(name));
}

SEA_EXPORT void itt_metadata_add(void* domain, uint64_t id, void* name, double value) {
    __itt_metadata_add(reinterpret_cast<__itt_domain*>(domain),
                       id ? __itt_id_make(domain, id) : __itt_null,
                       reinterpret_cast<__itt_string_handle*>(name),
                       __itt_metadata_double,
                       1,
                       &value);
}

SEA_EXPORT void itt_metadata_add_str(void* domain, uint64_t id, void* name, const char* value) {
    __itt_metadata_str_add(reinterpret_cast<__itt_domain*>(domain),
                           id ? __itt_id_make(domain, id) : __itt_null,
                           reinterpret_cast<__itt_string_handle*>(name),
                           value,
                           0);
}

SEA_EXPORT void itt_metadata_add_blob(void* domain, uint64_t id, void* name, const void* value, uint32_t size) {
    __itt_metadata_add(reinterpret_cast<__itt_domain*>(domain),
                       id ? __itt_id_make(domain, id) : __itt_null,
                       reinterpret_cast<__itt_string_handle*>(name),
                       __itt_metadata_unknown,
                       size,
                       const_cast<void*>(value));
}

SEA_EXPORT void itt_task_end(void* domain, uint64_t timestamp) {
    __itt_task_end_ex(reinterpret_cast<__itt_domain*>(domain), nullptr, timestamp);
}

SEA_EXPORT void itt_task_end_overlapped(void* domain, uint64_t timestamp, uint64_t taskid) {
    __itt_task_end_overlapped_ex(reinterpret_cast<__itt_domain*>(domain),
                                 nullptr,
                                 timestamp,
                                 __itt_id_make(domain, taskid));
}

SEA_EXPORT void* itt_counter_create(void* domain, void* name) {
    return __itt_counter_create_typed(reinterpret_cast<__itt_string_handle*>(name)->strA,
                                      reinterpret_cast<__itt_domain*>(domain)->nameA,
                                      __itt_metadata_u64);
}

SEA_EXPORT void itt_set_counter(void* id, double value, uint64_t timestamp) {
    __itt_counter_set_value_ex(reinterpret_cast<__itt_counter>(id), nullptr, timestamp, &value);
}

SEA_EXPORT void* itt_create_track(const char* group, const char* track) {
    return __itt_track_create(__itt_track_group_create(((group) ? __itt_string_handle_create(group) : nullptr),
                                                       __itt_track_group_type_normal),
                              __itt_string_handle_create(track),
                              __itt_track_type_normal);
}

SEA_EXPORT void itt_set_track(void* track) {
    __itt_set_track(reinterpret_cast<__itt_track*>(track));
}

SEA_EXPORT uint64_t itt_get_timestamp() {
    return (uint64_t)__itt_get_timestamp();
}

SEA_EXPORT void itt_write_time_sync_markers() {
#ifdef __linux__
    sea::WriteFTraceTimeSyncMarkers();
#endif
#ifdef __APPLE__
    sea::WriteKTraceTimeSyncMarkers();
#endif
}
};
