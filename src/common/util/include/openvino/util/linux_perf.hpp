#pragma once

#ifndef WIN32
/* echo 0 | sudo tee /proc/sys/kernel/perf_event_paranoid */
#define ENABLE_LINUX_PERF
#endif

#ifdef ENABLE_LINUX_PERF
#include <linux/perf_event.h>
#include <time.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/ioctl.h>
#include <iostream>
#include <string>

#if __GLIBC__ == 2 && __GLIBC_MINOR__ < 30
#include <sys/syscall.h>
#define gettid() syscall(SYS_gettid)
#endif

inline int perf_event_open(struct perf_event_attr *attr, pid_t pid, int cpu, int group_fd, unsigned long flags) {
	return syscall(__NR_perf_event_open, attr, pid, cpu, group_fd, flags);
}

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <atomic>
#include <x86intrin.h>
#include <sys/mman.h>
#include <thread>
#include <iostream>
#include <fstream>
#include <sstream>
#include <deque>
#include <mutex>
#include <set>
#include <iomanip>
#include <functional>
#include <limits>

namespace LinuxPerf {

#define _LINE_STRINGIZE(x) _LINE_STRINGIZE2(x)
#define _LINE_STRINGIZE2(x) #x
#define LINE_STRING _LINE_STRINGIZE(__LINE__)

#define LINUX_PERF_ "\e[33m[LINUX_PERF:" LINE_STRING "]\e[0m "

inline uint64_t get_time_ns() {
    struct timespec tp0;
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &tp0) != 0) {
        perror(LINUX_PERF_"clock_gettime(CLOCK_MONOTONIC_RAW,...) failed!");
        abort();
    }
    return (tp0.tv_sec * 1000000000) + tp0.tv_nsec;    
}

struct TscCounter {
    uint64_t tsc_ticks_per_second;
    uint64_t tsc_ticks_base;
    double tsc_to_usec(uint64_t tsc_ticks) const {
        if (tsc_ticks < tsc_ticks_base)
            return 0;
        return (tsc_ticks - tsc_ticks_base) * 1000000.0 / tsc_ticks_per_second;
    }
    double tsc_to_usec(uint64_t tsc_ticks0, uint64_t tsc_ticks1) const {
        if (tsc_ticks1 < tsc_ticks0)
            return 0;
        return (tsc_ticks1 - tsc_ticks0) * 1000000.0 / tsc_ticks_per_second;
    }
    TscCounter() {
        uint64_t start_ticks = __rdtsc();
        std::this_thread::sleep_for(std::chrono::seconds(1));
        tsc_ticks_per_second = (__rdtsc() - start_ticks);
        std::cout << LINUX_PERF_"tsc_ticks_per_second = " << tsc_ticks_per_second << std::endl;
        tsc_ticks_base = __rdtsc();

        // use CLOCK_MONOTONIC_RAW instead of TSC
        tsc_ticks_per_second = 1000000000; // ns
        tsc_ticks_base = get_time_ns();
    }
};

class IPerfEventDumper {
public:
    virtual void dump_json(std::ofstream& fw, TscCounter& tsc) = 0;
};

struct PerfEventJsonDumper {
    std::mutex g_mutex;
    std::set<IPerfEventDumper*> all_dumpers;
    const char* dump_file_name = "perf_dump.json";
    bool dump_file_over = false;
    bool not_finalized = true;
    std::ofstream fw;
    std::atomic_int totalProfilerManagers{0};
    TscCounter tsc;

    ~PerfEventJsonDumper() {
        if (not_finalized)
            finalize();
    }

    void finalize() {
        if (!not_finalized)
            return;
        std::lock_guard<std::mutex> guard(g_mutex);
        if (dump_file_over || all_dumpers.empty())
            return;

        // start dump
        fw.open(dump_file_name, std::ios::out);
        fw << "{\n";
        fw << "\"schemaVersion\": 1,\n";
        fw << "\"traceEvents\": [\n";
        fw.flush();

        for (auto& pthis : all_dumpers) {
            pthis->dump_json(fw, tsc);
        }
        all_dumpers.clear();

        fw << R"({
            "name": "Profiler End",
            "ph": "i",
            "s": "g",
            "pid": "Traces",
            "tid": "Trace OV Profiler",
            "ts":)"
           << tsc.tsc_to_usec(get_time_ns()) << "}",
            fw << "]\n";
        fw << "}\n";
        auto total_size = fw.tellp();
        fw.close();
        dump_file_over = true;
        not_finalized = false;

        std::cout << LINUX_PERF_"Dumpped ";
        
        if (total_size < 1024) std::cout << total_size << " bytes ";
        else if (total_size < 1024*1024) std::cout << total_size/1024 << " KB ";
        else std::cout << total_size/(1024 * 1024) << " MB ";
        std::cout << " to " << dump_file_name << std::endl;
    }

    int register_manager(IPerfEventDumper* pthis) {
        std::lock_guard<std::mutex> guard(g_mutex);
        std::stringstream ss;
        auto serial_id = totalProfilerManagers.fetch_add(1);
        ss << LINUX_PERF_"#" << serial_id << "(" << pthis << ") : is registed." << std::endl;
        std::cout << ss.str();
        all_dumpers.emplace(pthis);
        return serial_id;
    }

    static PerfEventJsonDumper& get() {
        static PerfEventJsonDumper inst;
        return inst;
    }
};

inline std::vector<std::string> str_split(const std::string& s, std::string delimiter) {
    std::vector<std::string> ret;
    size_t last = 0;
    size_t next = 0;
    while ((next = s.find(delimiter, last)) != std::string::npos) {
        //std::cout << last << "," << next << "=" << s.substr(last, next-last) << "\n";
        ret.push_back(s.substr(last, next-last));
        last = next + 1;
    }
    ret.push_back(s.substr(last));
    return ret;
}

template<typename T>
T& read_ring_buffer(perf_event_mmap_page& meta, uint64_t& offset) {
    auto offset0 = offset;
    offset += sizeof(T);
    return *reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(&meta) + meta.data_offset + (offset0)%meta.data_size);
}

struct PerfRawConfig {
    PerfRawConfig() {
        // env var defined raw events
        const char* str_raw_config = std::getenv("LINUX_PERF");
        if (str_raw_config) {
            CPU_ZERO(&cpu_mask);
            // options are separated by ":" as PATH
            auto options = str_split(str_raw_config, ":");
            for(auto& opt : options) {
                auto items = str_split(opt, "=");
                if (items.size() == 2) {
                    if (items[0] == "dump") {
                        // limit the number of dumps per thread
                        dump = strtoll(&items[1][0], nullptr, 0);
                    } else if (items[0] == "cpus") {
                        // thread's affinity (cpu-binding) can be changed by threading-libs(TBB/OpenMP) anytime
                        // sched_getaffinity() can only get correct binding at start-up time, another way is to specify it 
                        // also too many events may generate if per-thread event is used, cpus can limit
                        // cpus=56
                        // cpus=56.57.59
                        auto cpus = str_split(items[1], ",");
                        CPU_ZERO(&cpu_mask);
                        for(auto& cpu : cpus) {
                            CPU_SET(std::atoi(cpu.c_str()), &cpu_mask);
                        }
                    } else {
                        auto config = strtoul(&items[1][0], nullptr, 0);
                        if (config > 0)
                            raw_configs.emplace_back(items[0], config);
                    }
                }
                if (items.size() == 1) {
                    if (items[0] == "switch-cpu") {
                        // get cpu_mask as early as possible
                        switch_cpu = true;
                        CPU_ZERO(&cpu_mask);
                        if (sched_getaffinity(getpid(), sizeof(cpu_set_t), &cpu_mask)) {
                            perror(LINUX_PERF_"sched_getaffinity failed:");
                            abort();
                        }
                    }
                    if (items[0] == "dump")
                        dump = std::numeric_limits<int64_t>::max(); // no limit to number of dumps
                }
            }

            for(auto& cfg : raw_configs) {
                printf(LINUX_PERF_" config: %s=0x%lx\n", cfg.first.c_str(), cfg.second);
            }
            if (switch_cpu) {
                printf(LINUX_PERF_" config: switch_cpu\n");
            }
            if (dump)
                printf(LINUX_PERF_" config: dump=%ld\n", dump);
            if (CPU_COUNT(&cpu_mask)) {
                printf(LINUX_PERF_" config: cpus=");
                for (int cpu = 0; cpu < (int)sizeof(cpu_set_t)*8; cpu++)
                    if(CPU_ISSET(cpu, &cpu_mask)) printf("%d,", cpu);
                printf("\n");
            }
        } else {
            printf(LINUX_PERF_" LINUX_PERF is unset, example: LINUX_PERF=dump,switch-cpu,L2_MISS=0x10d1\n");
        }
    }

    bool dump_on_cpu(int cpu) {
        if (dump == 0)
            return false;
        if (CPU_COUNT(&cpu_mask))
            return CPU_ISSET(cpu, &cpu_mask);
        return true;
    }

    int64_t dump = 0;
    cpu_set_t cpu_mask;
    bool switch_cpu = false;
    std::vector<int> dump_cpus;
    std::vector<std::pair<std::string, uint64_t>> raw_configs;

    static PerfRawConfig& get() {
        static PerfRawConfig inst;
        return inst;
    }
};


// context switch events
// this will visualize 
struct PerfEventCtxSwitch : public IPerfEventDumper {
    bool is_enabled;

    struct event {
        int fd;
        perf_event_mmap_page * meta;
        int cpu;
        uint64_t ctx_switch_in_time;
        uint64_t ctx_switch_in_tid;
        uint64_t ctx_last_time;

        event(int fd, perf_event_mmap_page * meta): fd(fd), meta(meta) {}
    };
    std::vector<event> events;

    PerfEventCtxSwitch() {
        is_enabled = PerfRawConfig::get().switch_cpu;
        if (is_enabled) {
            // make sure TSC in PerfEventJsonDumper is the very first thing to initialize
            PerfEventJsonDumper::get().register_manager(this);

            // open fd for each CPU
            cpu_set_t mask = PerfRawConfig::get().cpu_mask;

            long number_of_processors = sysconf(_SC_NPROCESSORS_ONLN);
            printf(LINUX_PERF_"sizeof(cpu_set_t):%lu: _SC_NPROCESSORS_ONLN=%ld CPU_COUNT=%d\n", sizeof(cpu_set_t), number_of_processors, CPU_COUNT(&mask));
            if (CPU_COUNT(&mask) >= number_of_processors) {
                printf(LINUX_PERF_" no affinity is set, will not enable PerfEventCtxSwitch\n");
                is_enabled = false;
                return;
            }

            for (int cpu = 0; cpu < (int)sizeof(cpu_set_t)*8; cpu++) {
                auto is_set = CPU_ISSET(cpu, &mask);
                if (!is_set) continue;

                perf_event_attr pea;
                memset(&pea, 0, sizeof(struct perf_event_attr));
                pea.type = PERF_TYPE_HARDWARE;
                pea.size = sizeof(struct perf_event_attr);
                pea.config = PERF_COUNT_HW_REF_CPU_CYCLES;  // not the point, can be any
                pea.disabled = 0;
                pea.exclude_kernel = 1;
                pea.exclude_hv = 1;
                pea.read_format = PERF_FORMAT_GROUP | PERF_FORMAT_ID;        
                // pinned: It applies only to hardware counters and only to group leaders
                pea.pinned = 1;
                pea.read_format |= PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;

                // for group master, generate PERF_RECORD_SWITCH into ring-buffer
                // is helpful to visualize context switch
                pea.context_switch = 1;
                // then TID, TIME, ID, STREAM_ID, and CPU can additionally be included in non-PERF_RECORD_SAMPLEs
                // if the  corresponding sample_type is selected
                pea.sample_id_all = 1;
                pea.sample_type = PERF_SAMPLE_TIME | PERF_SAMPLE_TID | PERF_SAMPLE_CPU;
                auto mmap_length = sysconf(_SC_PAGESIZE) * (1024 + 1);
                pea.use_clockid = 1;
                pea.clockid = CLOCK_MONOTONIC_RAW;

                // calling thread on any processor
                pid_t pid = -1;
                // measures all processes/threads on the specified CPU
                int ctx_switch_fd = perf_event_open(&pea, pid, cpu, -1, 0);
                if (ctx_switch_fd < 0) {
                    perror(LINUX_PERF_"PerfEventCtxSwitch perf_event_open failed (check /proc/sys/kernel/perf_event_paranoid please)");
                    abort();
                }

                auto* ctx_switch_pmeta = reinterpret_cast<perf_event_mmap_page*>(mmap(NULL, mmap_length, PROT_READ | PROT_WRITE, MAP_SHARED, ctx_switch_fd, 0));
                if (ctx_switch_pmeta == MAP_FAILED) {
                    perror(LINUX_PERF_"mmap perf_event_mmap_page failed:");
                    close(ctx_switch_fd);
                    abort();
                }
                printf(LINUX_PERF_"perf_event_open CPU_WIDE context_switch on cpu %d, ctx_switch_fd=%d\n", cpu, ctx_switch_fd);
                events.emplace_back(ctx_switch_fd, ctx_switch_pmeta);
                events.back().ctx_switch_in_time = get_time_ns();
                events.back().ctx_last_time = get_time_ns();
                events.back().cpu = cpu;
            }
            my_pid = getpid();
            my_tid = gettid();
        }
    }

    ~PerfEventCtxSwitch() {
        if (is_enabled) {
            PerfEventJsonDumper::get().finalize();
        }
        for(auto& ev : events) {
            close(ev.fd);
        }
    }

    struct ProfileData {
        uint64_t tsc_start;
        uint64_t tsc_end;
        uint32_t tid;
        uint32_t cpu;
        bool preempt;   // preempt means current TID preempts previous thread
    };

    std::deque<ProfileData> all_dump_data;

    void dump_json(std::ofstream& fw, TscCounter& tsc) override {
        static std::atomic_uint64_t async_evid{0};
        if (!is_enabled) return;

        updateRingBuffer();

        auto data_size = all_dump_data.size();
        if (!data_size) return;

        for (auto& ev : events) {
            if (ev.ctx_switch_in_time == 0) continue;
            all_dump_data.emplace_back();
            auto* pd = &all_dump_data.back();
            pd->tid = ev.ctx_switch_in_tid;
            pd->cpu = ev.cpu;
            pd->tsc_start = ev.ctx_switch_in_time;
            pd->tsc_end = get_time_ns();
            ev.ctx_switch_in_time = 0;
        }

        auto pid = 9999;    // fake pid for CPU
        auto cat = "TID";
        
        // TID is used for CPU id instead
        for (auto& d : all_dump_data) {
            auto duration = tsc.tsc_to_usec(d.tsc_start, d.tsc_end);
            auto start = tsc.tsc_to_usec(d.tsc_start);
            //auto end = tsc.tsc_to_usec(d.tsc_end);
            auto cpu_id = d.cpu;

            fw << "{\"ph\": \"X\", \"name\": \"" << d.tid << "\", \"cat\":\"" << cat << "\","
                << "\"pid\": " << pid << ", \"tid\": \"CPU" << cpu_id <<  "\","
                << "\"ts\": " << std::setprecision (15) << start << ", \"dur\": " << duration << "},\n";
        }
    }

    bool ring_buffer_verbose = false;
    uint32_t my_pid = 0;
    uint32_t my_tid = 0;
    std::atomic<int> atom_gard{0};

    void updateRingBuffer() {
        // only one thread can enter
        const int lock_value = atom_gard.exchange(1);
        if (lock_value == 1) {
            // has been locked, return;
            return;
        }

        // only update when any ring-buffer is half loaded
        bool need_update = false;
        for(auto& ev : events) {
            auto& mmap_meta = *ev.meta;
            auto used_size = (mmap_meta.data_tail - mmap_meta.data_head) % mmap_meta.data_size;
            if (used_size > (mmap_meta.data_size >> 1)) {
                need_update = true;
                break;
            }
        }

        if (!need_update) {
            // unlock
            atom_gard.exchange(0);
            return;
        }

        for(auto& ev : events) {
            auto& mmap_meta = *ev.meta;
            uint64_t head0 = mmap_meta.data_tail;
            uint64_t head1 = mmap_meta.data_head;
            //printf("ring-buffer@end: %lu~%lu %llu %llu %llu\n", head0, head1, group_meta.data_tail, group_meta.data_offset, group_meta.data_size);

            if (head0 != head1) {
                if (ring_buffer_verbose) {
                    printf("PERF_RECORD_SWITCH = %d\n", PERF_RECORD_SWITCH);
                    printf("PERF_RECORD_SWITCH_CPU_WIDE = %d\n", PERF_RECORD_SWITCH_CPU_WIDE);
                    printf("PERF_RECORD_MISC_SWITCH_OUT = %d\n", PERF_RECORD_MISC_SWITCH_OUT);
                    printf("PERF_RECORD_MISC_SWITCH_OUT_PREEMPT  = %d\n", PERF_RECORD_MISC_SWITCH_OUT_PREEMPT);
                }

                while(head0 < head1) {
                    auto h0 = head0;
                    auto type = read_ring_buffer<__u32>(mmap_meta, head0);
                    auto misc = read_ring_buffer<__u16>(mmap_meta, head0);
                    auto size = read_ring_buffer<__u16>(mmap_meta, head0);
                    uint32_t next_prev_pid = 0, next_prev_tid = 0;
                    if (type == PERF_RECORD_SWITCH_CPU_WIDE) {
                        // previous PID/TID if switching-in
                        // next PID/TID if switching-out
                        next_prev_pid = read_ring_buffer<__u32>(mmap_meta, head0);
                        next_prev_tid = read_ring_buffer<__u32>(mmap_meta, head0);
                    }
                    auto pid = read_ring_buffer<__u32>(mmap_meta, head0);
                    auto tid = read_ring_buffer<__u32>(mmap_meta, head0);
                    auto time = read_ring_buffer<uint64_t>(mmap_meta, head0);
                    auto cpu = read_ring_buffer<__u32>(mmap_meta, head0);
                    auto reserved0 = read_ring_buffer<__u32>(mmap_meta, head0);
                    (void)reserved0;
                    (void)next_prev_pid;
                    (void)pid;

                    // skip idle process (with TID 0)
                    if (tid > 0 && ring_buffer_verbose) {
                        printf("event: %lu/%lu\ttype,misc,size=(%u,%u,%u) cpu%u,next_prev_tid=%u,tid=%u  time:(%lu), (+%lu)\n",
                            h0, head1,
                            type, misc, size,
                            cpu, next_prev_tid, tid,
                            time,
                            time - ev.ctx_last_time);
                    }

                    if (type == PERF_RECORD_SWITCH_CPU_WIDE && tid > 0) {
                        if (misc & PERF_RECORD_MISC_SWITCH_OUT || misc & PERF_RECORD_MISC_SWITCH_OUT_PREEMPT) {
                            // switch out
                            // generate a log
                            all_dump_data.emplace_back();
                            auto* pd = &all_dump_data.back();
                            pd->tid = tid;
                            pd->cpu = cpu;
                            pd->preempt = (misc & PERF_RECORD_MISC_SWITCH_OUT_PREEMPT);
                            //printf("ctx_switch_in_time=%lu\n", ctx_switch_in_time);
                            pd->tsc_start = ev.ctx_switch_in_time;
                            pd->tsc_end = time;

                            if (ring_buffer_verbose) printf("\t  cpu: %u tid: %u  %lu (+%lu)\n", cpu, tid, ev.ctx_switch_in_time, time-ev.ctx_switch_in_time);

                            ev.ctx_switch_in_time = 0;
                        } else {
                            // switch in
                            ev.ctx_switch_in_time = time;
                            ev.ctx_switch_in_tid = tid;
                        }
                    }

                    ev.ctx_last_time = time;
                    head0 += size - (head0 - h0);
                }

                if (head0 != head1) {
                    printf("head0(%lu) != head1(%lu)\n", head0, head1);
                    abort();
                }

                // update tail so kernel can keep generate event records
                mmap_meta.data_tail = head0;
                std::atomic_thread_fence(std::memory_order_seq_cst);
            }
        }
        atom_gard.exchange(0);
    }

    static PerfEventCtxSwitch& get() {
        static PerfEventCtxSwitch inst;
        return inst;
    }
};

/*
RAW HARDWARE EVENT DESCRIPTOR
       Even when an event is not available in a symbolic form within perf right now, it can be encoded in a per processor specific way.

       For instance For x86 CPUs NNN represents the raw register encoding with the layout of IA32_PERFEVTSELx MSRs (see [Intel® 64 and IA-32 Architectures Software Developer’s Manual Volume 3B: System Programming Guide] Figure 30-1
       Layout of IA32_PERFEVTSELx MSRs) or AMD’s PerfEvtSeln (see [AMD64 Architecture Programmer’s Manual Volume 2: System Programming], Page 344, Figure 13-7 Performance Event-Select Register (PerfEvtSeln)).

       Note: Only the following bit fields can be set in x86 counter registers: event, umask, edge, inv, cmask. Esp. guest/host only and OS/user mode flags must be setup using EVENT MODIFIERS.

 event 7:0
 umask 15:8
 edge  18
 inv   23
 cmask 31:24
*/
#define X86_RAW_EVENT(EventSel, UMask, CMask) ((CMask << 24) | (UMask << 8) | (EventSel))

struct PerfEventGroup : public IPerfEventDumper {
    int group_fd = -1;
    uint64_t read_format;

    struct event {
        int fd = -1;
        uint64_t id = 0;
        uint64_t pmc_index = 0;
        perf_event_mmap_page* pmeta = nullptr;
        std::string name = "?";
        char format[32];
    };
    std::vector<event> events;

    uint64_t read_buf[512]; // 4KB
    uint64_t time_enabled;
    uint64_t time_running;
    uint64_t pmc_width;
    uint64_t pmc_mask;
    uint64_t values[32];
    uint32_t tsc_time_shift;
    uint32_t tsc_time_mult;

    // ref_cpu_cycles even id
    // this event is fixed function counter provided by most x86 CPU
    // and it provides TSC clock which is:
    //    - very high-resolution (<1ns or >1GHz)
    //    - independent of CPU-frequency throttling
    int ref_cpu_cycles_evid = -1;
    int sw_task_clock_evid = -1;
    int hw_cpu_cycles_evid = -1;
    int hw_instructions_evid = -1;

    struct ProfileData {
        uint64_t tsc_start;
        uint64_t tsc_end;
        std::string title;
        const char * cat;
        int32_t id;
        static const int data_size = 16; // 4(fixed) + 8(PMU) + 4(software)
        uint64_t data[data_size] = {0};
        // f/i/u/p
        char extra_data_type[data_size] = {0};
        union {
            double f;
            int64_t i;
            void * p;
        } extra_data[data_size];

        template<typename T>
        char get_extra_type(T t) {
            if (std::is_pointer<T>::value) return 'p';
            if (std::is_floating_point<T>::value) return 'f';
            if (std::is_integral<T>::value) return 'i';
            return '\0';
        }
        template<typename T>
        void set_extra_data(int i, T* t) { extra_data[i].p = t; }
        void set_extra_data(int i, float t) { extra_data[i].f = t; }
        void set_extra_data(int i, double t) { extra_data[i].f = t; }
        template<typename T>
        void set_extra_data(int i, T t) {
            static_assert(std::is_integral<T>::value);
            extra_data[i].i = t;
        }

        template <typename ... Values>
        void set_all_extra_data(Values... vals) {
            static_assert(data_size >= sizeof...(vals));
            int j = 0;
            int unused1[] = { 0, (set_extra_data(j++, vals), 0)... };
            (void)unused1;
            j = 0;
            int unused2[] = { 0, (extra_data_type[j++] = get_extra_type(vals), 0)... };
            (void)unused2;
            extra_data_type[j] = '\0';
        }

        ProfileData(const std::string& title) : title(title) {
            start();
        }
        void start() {
            tsc_start = get_time_ns();
        }
        void stop() {
            tsc_end = get_time_ns();
        }
    };

    bool enable_dump_json = false;
    int64_t dump_limit = 0;
    std::deque<ProfileData> all_dump_data;
    int serial;

    using CallBackEventArgsSerializer = std::function<void(std::ostream& fw, double usec, uint64_t* counters)>;
    CallBackEventArgsSerializer fn_evt_args_serializer;

    void dump_json(std::ofstream& fw, TscCounter& tsc) override {
        static std::atomic_uint64_t async_evid{0};
        if (!enable_dump_json)
            return;
        auto data_size = all_dump_data.size();
        if (!data_size)
            return;

        for (auto& d : all_dump_data) {
            auto duration = tsc.tsc_to_usec(d.tsc_start, d.tsc_end);
            auto title = std::string(d.title) + "_" + std::to_string(d.id);
            auto cat = d.cat;
            //auto pid = serial;
            auto start = tsc.tsc_to_usec(d.tsc_start);
            //auto end = tsc.tsc_to_usec(d.tsc_end);

            if (d.id < 0) {
                // async events
                // {"cat": "foo", "name": "async_read2", "pid": 4092243, "id": 4092246, "ph": "b", "ts": 23819.718},
                fw << "{\"ph\": \"b\", \"name\": \"" << d.title << "\", \"cat\":\"" << cat << "\","
                    << "\"pid\": " << my_pid << ", \"id\": " << (-d.id) << ","
                    << "\"ts\": " << std::setprecision (15) << start << "},";

                fw << "{\"ph\": \"e\", \"name\": \"" << d.title << "\", \"cat\":\"" << cat << "\","
                    << "\"pid\": " << my_pid << ", \"id\": " << (-d.id) << ","
                    << "\"ts\": " << std::setprecision (15) << tsc.tsc_to_usec(d.tsc_end) << ",";
            } else {
                fw << "{\"ph\": \"X\", \"name\": \"" << title << "\", \"cat\":\"" << cat << "\","
                    << "\"pid\": " << my_pid << ", \"tid\": " << my_tid << ","
                    << "\"ts\": " << std::setprecision (15) << start << ", \"dur\": " << duration << ",";
            }

            fw << "\"args\":{";
            {
                std::stringstream ss;
                if (fn_evt_args_serializer)
                    fn_evt_args_serializer(ss, duration, d.data);
                if (sw_task_clock_evid >= 0) {
                    // PERF_COUNT_SW_TASK_CLOCK in nano-seconds
                    ss << "\"CPU Usage\":" << (d.data[sw_task_clock_evid] * 1e-3)/duration << ",";
                }
                if (hw_cpu_cycles_evid >= 0) {
                    if (sw_task_clock_evid >= 0 && d.data[sw_task_clock_evid] > 0) {
                        ss << "\"CPU Freq(GHz)\":" << static_cast<double>(d.data[hw_cpu_cycles_evid])/d.data[sw_task_clock_evid] << ",";
                    } else {
                        ss << "\"CPU Freq(GHz)\":" << static_cast<double>(d.data[hw_cpu_cycles_evid])*1e-3/duration << ",";
                    }
                    if (hw_instructions_evid >= 0 && d.data[hw_instructions_evid] > 0) {
                        ss << "\"CPI\":" << static_cast<double>(d.data[hw_cpu_cycles_evid])/d.data[hw_instructions_evid] << ",";
                    }
                }
                auto prev_locale = ss.imbue(std::locale(""));
                const char * sep = "";
                for(size_t i = 0; i < events.size() && i < d.data_size; i++) {
                    ss << sep << "\"" << events[i].name << "\":\"" << d.data[i] << "\"";
                    sep = ",";
                }
                ss.imbue(prev_locale);
                if (d.extra_data_type[0] != 0) {
                    sep = "";
                    ss << ",\"Extra Data\":[";
                    for(size_t i = 0; i < d.data_size && (d.extra_data_type[i] != 0); i++) {
                        if (d.extra_data_type[i] == 'f') ss << sep << d.extra_data[i].f;
                        else if (d.extra_data_type[i] == 'i') ss << sep << d.extra_data[i].i;
                        else if (d.extra_data_type[i] == 'p') ss << sep << "\"" << d.extra_data[i].p << "\"";
                        else ss << sep << "\"?\"";
                        sep = ",";
                    }
                    ss << "]";
                }
                fw << ss.str();
            }
            fw << "}},\n";
        }
        all_dump_data.clear();
        std::cout << LINUX_PERF_"#" << serial << "(" << this << ") finalize: dumpped " << data_size << std::endl;
    }

    uint64_t operator[](size_t i) {
        if (i < events.size()) {
            return values[i];
        } else {
            printf(LINUX_PERF_"PerfEventGroup: operator[] with index %lu oveflow (>%lu)\n", i, events.size());
            abort();
        }
        return 0;
    }
    
    PerfEventGroup() = default;

    struct Config {
        uint32_t type;
        uint64_t config;
        const char * name;
        Config(uint32_t type, uint64_t config, const char * name = "?") : type(type), config(config), name(name) {}
    };

    uint32_t my_pid = 0;
    uint32_t my_tid = 0;

    PerfEventGroup(const std::vector<Config> type_configs, CallBackEventArgsSerializer fn = {}) : fn_evt_args_serializer(fn) {
        for(auto& tc : type_configs) {
            if (tc.type == PERF_TYPE_SOFTWARE) {
                add_sw(tc.config);
            }
            if (tc.type == PERF_TYPE_HARDWARE) {
                add_hw(tc.config);
            }
            if (tc.type == PERF_TYPE_RAW) {
                add_raw(tc.config);
            }
            events.back().name = tc.name;
            snprintf(events.back().format, sizeof(events.back().format), "%%%lulu, ", strlen(tc.name));
        }

        // env var defined raw events
        for (auto raw_cfg : PerfRawConfig::get().raw_configs) {
            add_raw(raw_cfg.second);
            events.back().name = raw_cfg.first;
        }

        dump_limit = PerfRawConfig::get().dump;
        enable_dump_json = PerfRawConfig::get().dump_on_cpu(sched_getcpu());
        serial = 0;
        if (enable_dump_json) {
            serial = PerfEventJsonDumper::get().register_manager(this);
        }
        my_pid = getpid();
        my_tid = gettid();

        enable();
    }

    ~PerfEventGroup() {
        if (enable_dump_json)
            PerfEventJsonDumper::get().finalize();
        disable();
        for(auto & ev : events) {
            close(ev.fd);
        }
    }

    void show_header() {
        std::stringstream ss;
        ss << "\e[33m";
        ss << "#" << serial << ":";
        for(auto& ev : events) {
            ss << ev.name << ", ";
        }
        ss << "\e[0m\n";
        std::cout << ss.str();
    }

    void add_raw(uint64_t config, bool pinned=false) {
        perf_event_attr pea;
        memset(&pea, 0, sizeof(struct perf_event_attr));
        pea.type = PERF_TYPE_RAW;
        pea.size = sizeof(struct perf_event_attr);
        pea.config = config;
        pea.disabled = 1;
        pea.exclude_kernel = 1;
        pea.exclude_hv = 1;
        pea.read_format = PERF_FORMAT_GROUP | PERF_FORMAT_ID;        
        if (pinned && group_fd == -1) {
            // pinned: It applies only to hardware counters and only to group leaders
            pea.pinned = 1;
        }
        if (group_fd == -1) {
            pea.read_format |= PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
        }
        add(&pea);
    }

    void add_hw(uint64_t config, bool pinned=false) {
        perf_event_attr pea;
        memset(&pea, 0, sizeof(struct perf_event_attr));
        pea.type = PERF_TYPE_HARDWARE;
        pea.size = sizeof(struct perf_event_attr);
        pea.config = config;
        pea.disabled = 1;
        pea.exclude_kernel = 1;
        pea.exclude_hv = 1;
        pea.read_format = PERF_FORMAT_GROUP | PERF_FORMAT_ID;        
        if (pinned && group_fd == -1) {
            // pinned: It applies only to hardware counters and only to group leaders
            pea.pinned = 1;
        }
        if (group_fd == -1) {
            pea.read_format |= PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
        }
        add(&pea);
    }

    void add_sw(uint64_t config) {
        perf_event_attr pea;
        memset(&pea, 0, sizeof(struct perf_event_attr));
        pea.type = PERF_TYPE_SOFTWARE;
        pea.size = sizeof(struct perf_event_attr);
        pea.config = config;
        pea.disabled = 1;
        pea.exclude_kernel = 0; // some SW events are counted as kernel
        pea.exclude_hv = 1;
        //pea.pinned = 1;   //sw event cannot set pinned!!!
        pea.read_format = PERF_FORMAT_GROUP | PERF_FORMAT_ID ;
        add(&pea);
    }

    void add(perf_event_attr* pev_attr, pid_t pid = 0, int cpu = -1) {
        event ev;

        size_t mmap_length = sysconf(_SC_PAGESIZE) * 1;
        // clockid must consistent within group
        pev_attr->use_clockid = 1;
        // can be synched with clock_gettime(CLOCK_MONOTONIC_RAW)
        pev_attr->clockid = CLOCK_MONOTONIC_RAW;

        RETRY:
        ev.fd = perf_event_open(pev_attr, pid, cpu, group_fd, 0);
        if (ev.fd < 0) {
            if (!pev_attr->exclude_kernel) {
                printf(LINUX_PERF_"perf_event_open(type=%d,config=%lld) with exclude_kernel=0 failed (due to /proc/sys/kernel/perf_event_paranoid is 2),  set exclude_kernel=1 and retry...\n",
                       pev_attr->type, pev_attr->config);
                pev_attr->exclude_kernel = 1;
                goto RETRY;
            } else {
                printf(LINUX_PERF_"perf_event_open(type=%d,config=%lld) failed", pev_attr->type, pev_attr->config);
                perror("");
                abort();
            }
        }
        ioctl(ev.fd, PERF_EVENT_IOC_ID, &ev.id);

        ev.pmeta = reinterpret_cast<perf_event_mmap_page*>(mmap(NULL, mmap_length, PROT_READ | PROT_WRITE, MAP_SHARED, ev.fd, 0));
        if (ev.pmeta == MAP_FAILED) {
            perror(LINUX_PERF_"mmap perf_event_mmap_page failed:");
            close(ev.fd);
            abort();
        }

        if (group_fd == -1) {
            group_fd = ev.fd;
            read_format = pev_attr->read_format;
        }
        if (pev_attr->type == PERF_TYPE_HARDWARE && pev_attr->config == PERF_COUNT_HW_REF_CPU_CYCLES) {
            ref_cpu_cycles_evid = events.size();
        }
        if (pev_attr->type == PERF_TYPE_SOFTWARE && pev_attr->config == PERF_COUNT_SW_TASK_CLOCK) {
            sw_task_clock_evid = events.size();
        }
        if (pev_attr->type == PERF_TYPE_HARDWARE && pev_attr->config == PERF_COUNT_HW_CPU_CYCLES) {
            hw_cpu_cycles_evid = events.size();
        }
        if (pev_attr->type == PERF_TYPE_HARDWARE && pev_attr->config == PERF_COUNT_HW_INSTRUCTIONS) {
            hw_instructions_evid = events.size();
        }
        //printf("perf_event_open : fd=%d, id=%lu\n", ev.fd, ev.id);

        events.push_back(ev);
    }

    bool event_group_enabled = false;
    uint32_t num_events_no_pmc;

    void enable() {
        if (event_group_enabled)
            return;
        ioctl(group_fd, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP);
        ioctl(group_fd, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
        // PMC index is only valid when being enabled
        num_events_no_pmc = 0;
        for(auto& ev : events) {
            if (ev.pmc_index == 0 && ev.pmeta->cap_user_rdpmc) {
                uint32_t seqlock;
                do {
                    seqlock = ev.pmeta->lock;
                    std::atomic_thread_fence(std::memory_order_seq_cst);
                    ev.pmc_index = ev.pmeta->index;
                    pmc_width = ev.pmeta->pmc_width;
                    pmc_mask = 1;
                    pmc_mask = (pmc_mask << pmc_width) - 1;
                    if (ev.pmeta->cap_user_time) {
                        tsc_time_shift = ev.pmeta->time_shift;
                        tsc_time_mult = ev.pmeta->time_mult;
                        //printf("time: %u,%u\n", tsc_time_shift, tsc_time_mult);
                    }
                    std::atomic_thread_fence(std::memory_order_seq_cst);
                } while (ev.pmeta->lock != seqlock || (seqlock & 1));
            }
            // some events like PERF_TYPE_SOFTWARE cannot read using rdpmc()
            if (ev.pmc_index == 0)
                num_events_no_pmc ++;
        }
        event_group_enabled = true;
    }

    uint64_t tsc2nano(uint64_t cyc) {
        uint64_t quot, rem;
        quot  = cyc >> tsc_time_shift;
        rem   = cyc & (((uint64_t)1 << tsc_time_shift) - 1);
        return quot * tsc_time_mult + ((rem * tsc_time_mult) >> tsc_time_shift);
    }

    void disable() {
        if (!event_group_enabled)
            return;

        ioctl(group_fd, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);

        for(auto& ev : events) {
            ev.pmc_index = 0;
        }
        event_group_enabled = false;
    }

    uint64_t rdpmc(int i, uint64_t base = 0) {
        return (_rdpmc(events[i].pmc_index - 1) - base) & pmc_mask;
    }

    template<class FN>
    std::vector<uint64_t> rdpmc(FN fn, std::string name = {}, int64_t loop_cnt = 0, std::function<void(uint64_t, uint64_t*, char*&)> addinfo = {}) {
        int cnt = events.size();
        std::vector<uint64_t> pmc(cnt, 0);

        bool use_pmc = (num_events_no_pmc == 0);
        if (use_pmc) {
            for(int i = 0; i < cnt; i++) {
                if (events[i].pmc_index)
                    pmc[i] = _rdpmc(events[i].pmc_index - 1);
                else
                    pmc[i] = 0;
            }
        } else {
            read();
            for(int i = 0; i < cnt; i++) {
                pmc[i] = values[i];
            }
        }

        auto tsc0 = __rdtsc();
        fn();
        auto tsc1 = __rdtsc();

        if (use_pmc) {
            for(int i = 0; i < cnt; i++) {
                if (events[i].pmc_index)
                    pmc[i] = (_rdpmc(events[i].pmc_index - 1) - pmc[i]) & pmc_mask;
                else
                    pmc[i] = 0;
            }
        } else {
            read();
            for(int i = 0; i < cnt; i++) {
                pmc[i] -= values[i];
            }
        }

        if (!name.empty()) {
            char log_buff[1024];
            char * log = log_buff;
            log += sprintf(log, "\e[33m");
            for(int i = 0; i < cnt; i++) {
                log += sprintf(log, events[i].format, pmc[i]);
            }
            auto duration_ns = tsc2nano(tsc1 - tsc0);
            
            log += sprintf(log, "\e[0m [%16s] %.3f us", name.c_str(), duration_ns/1e3);
            if (hw_cpu_cycles_evid >= 0) {
                log += sprintf(log, " CPU:%.2f(GHz)", 1.0 * pmc[hw_cpu_cycles_evid] / duration_ns);
                if (hw_instructions_evid >= 0) {
                    log += sprintf(log, " CPI:%.2f", 1.0 * pmc[hw_cpu_cycles_evid] / pmc[hw_instructions_evid]);
                }
                if (loop_cnt > 0) {
                    // cycles per kernel (or per-iteration)
                    log += sprintf(log, " CPK:%.1fx%d", 1.0 * pmc[hw_cpu_cycles_evid] / loop_cnt, loop_cnt);
                }
            }
            if (addinfo) {
                addinfo(duration_ns, &pmc[0], log);
            }
            log += sprintf(log, "\n");
            printf(log_buff);
        }
        return pmc;
    }

    void read(bool verbose = false) {
        for(size_t i = 0; i < events.size(); i++) values[i] = 0;

        if (::read(group_fd, read_buf, sizeof(read_buf)) == -1) {
            perror(LINUX_PERF_"read perf event failed:");
            abort();
        }

        uint64_t * readv = read_buf;
        auto nr = *readv++;
        if (verbose) printf("number of counters:\t%lu\n", nr);
        time_enabled = 0;
        time_running = 0;
        if (read_format & PERF_FORMAT_TOTAL_TIME_ENABLED) {
            time_enabled = *readv++;
            if (verbose) printf("time_enabled:\t%lu\n", time_enabled);
        }
        if (read_format & PERF_FORMAT_TOTAL_TIME_RUNNING) {
            time_running = *readv++;
            if (verbose) printf("time_running:\t%lu\n", time_running);
        }

        for (size_t i = 0; i < nr; i++) {
            auto value = *readv++;
            auto id = *readv++;
            for (size_t k = 0; k < events.size(); k++) {
                if (id == events[k].id) {
                    values[k] = value;
                }
            }
        }

        if (verbose) {
            for (size_t k = 0; k < events.size(); k++) {
                printf("\t[%lu]: %lu\n", k, values[k]);
            }
        }
    }

    //================================================================================
    // profiler API with json_dump capability
    struct ProfileScope {
        PerfEventGroup* pevg = nullptr;
        ProfileData* pd = nullptr;
        bool do_unlock = false;
        ProfileScope() = default;
        ProfileScope(PerfEventGroup* pevg, ProfileData* pd, bool do_unlock = false) : pevg(pevg), pd(pd), do_unlock(do_unlock) {}

        // Move only
        ProfileScope(const ProfileScope&) = delete;
        ProfileScope& operator=(const ProfileScope&) = delete;

        ProfileScope(ProfileScope&& other) {
            pevg = other.pevg;
            pd = other.pd;
            other.pevg = nullptr;
            other.pd = nullptr;
        }

        ProfileScope& operator=(ProfileScope&& other) {
            if (&other != this) {
                pevg = other.pevg;
                pd = other.pd;
                other.pevg = nullptr;
                other.pd = nullptr;
            }

            return *this;
        }

        uint64_t* finish() {
            if (do_unlock) {
                PerfEventGroup::get_sampling_lock() --;
            }
            if (!pevg || !pd)
                return nullptr;

            pd->stop();
            bool use_pmc = (pevg->num_events_no_pmc == 0);
            if (use_pmc) {
                for (size_t i =0; i < pevg->events.size() && i < pd->data_size; i++)
                    if (pevg->events[i].pmc_index)
                        pd->data[i] = (_rdpmc(pevg->events[i].pmc_index - 1) - pd->data[i]) & pevg->pmc_mask;
                    else
                        pd->data[i] = 0;
            } else {
                pevg->read();
                for (size_t i =0; i < pevg->events.size() && i < pd->data_size; i++)
                    pd->data[i] = pevg->values[i] - pd->data[i];
            }
            pevg = nullptr;
            return pd->data;
        }

        ~ProfileScope() {
            finish();
        }
    };

    ProfileData* _profile(const std::string& title, int id = 0) {
        if (get_sampling_lock().load() != 0)
            return nullptr;
        if (dump_limit == 0)
            return nullptr;
        dump_limit --;

        PerfEventCtxSwitch::get().updateRingBuffer();

        all_dump_data.emplace_back(title);
        auto* pd = &all_dump_data.back();
        pd->cat = "enable";
        pd->id = id;

        // use rdpmc if possible
        bool use_pmc = (num_events_no_pmc == 0);
        if (use_pmc) {
            for (size_t i =0; i < events.size() && i < pd->data_size; i++)
                if (events[i].pmc_index)
                    pd->data[i] = _rdpmc(events[i].pmc_index - 1);
        } else {
            read();
            for (size_t i =0; i < events.size() && i < pd->data_size; i++)
                pd->data[i] = values[i];
        }

        return pd;
    }

    static PerfEventGroup& get() {
        thread_local PerfEventGroup pevg({
            {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, "HW_CPU_CYCLES"},
            {PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS, "HW_INSTRUCTIONS"},
            {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES, "HW_CACHE_MISSES"},
            //{PERF_TYPE_HARDWARE, PERF_COUNT_HW_REF_CPU_CYCLES, "HW_REF_CPU_CYCLES"},
            {PERF_TYPE_SOFTWARE, PERF_COUNT_SW_CONTEXT_SWITCHES, "SW_CONTEXT_SWITCHES"},
            // {PERF_TYPE_SOFTWARE, PERF_COUNT_SW_TASK_CLOCK, "SW_TASK_CLOCK"},
            {PERF_TYPE_SOFTWARE, PERF_COUNT_SW_PAGE_FAULTS, "SW_PAGE_FAULTS"},

            // XSNP_NONE                : ... were hits in L3 without snoops required                (data is not owned by any other core's local cache)
            // XSNP_FWD   /XSNP_HITM    : ... were HitM responses from shared L3                     (data was exclusivly/dirty owned by another core's local cache)
            // XSNP_NO_FWD/XSNP_HIT     : ... were L3 and cross-core snoop hits in on-pkg core cache (data was shared/clean in another core's local cache)

            // {PERF_TYPE_RAW, X86_RAW_EVENT(0xd2, 0x01, 0x00), "XSNP_MISS"},
            // {PERF_TYPE_RAW, X86_RAW_EVENT(0xd2, 0x02, 0x00), "XSNP_NO_FWD"},
            // {PERF_TYPE_RAW, X86_RAW_EVENT(0xd2, 0x04, 0x00), "XSNP_FWD"},
            // {PERF_TYPE_RAW, X86_RAW_EVENT(0xd2, 0x08, 0x00), "XSNP_NONE"},              
        });
        return pevg;
    }

    // this lock is global, affect all threads
    static std::atomic_int& get_sampling_lock() {
        static std::atomic_int sampling_lock{0};
        return sampling_lock;
    }
};

using ProfileScope = PerfEventGroup::ProfileScope;

// pwe-thread event group with default events pre-selected
template <typename ... Args>
ProfileScope Profile(const std::string& title, int id = 0, Args&&... args) {
    auto& pevg = PerfEventGroup::get();
    auto* pd = pevg._profile(title, id);
    if (pd) {
        pd->set_all_extra_data(std::forward<Args>(args)...);
    }
    return {&pevg, pd};
}

// overload accept sampling_probability, which can be used to disable profile in scope 
template <typename ... Args>
ProfileScope Profile(float sampling_probability, const std::string& title, int id = 0, Args&&... args) {
    auto& pevg = PerfEventGroup::get();
    auto* pd = pevg._profile(title, id);
    if (pd) {
        pd->set_all_extra_data(std::forward<Args>(args)...);
    }

    bool disable_profile = ((std::rand() % 1000)*0.001f >= sampling_probability);
    if (disable_profile) {
        PerfEventGroup::get_sampling_lock() ++;
    }
    return {&pevg, pd, disable_profile};
}

inline int Init() {
    // this is for capture all context switching events
    PerfEventCtxSwitch::get();

    // this is for making main threads the first process
    auto dummy = Profile("start");
    return 0;
}
} // namespace LinuxPerf

#define LINUX_PERF_LOG(...)  \
    auto _perf_ = LinuxPerf::Profile(__VA_ARGS__);

#else

#define LINUX_PERF_LOG(...)

#endif
