// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>
#include <atomic>
#include <map>
#include <queue>
#include <memory>
#include <climits>
#include <cpp_interfaces/impl/ie_infer_request_internal.hpp>
#include <cpp_interfaces/ie_task_executor.hpp>
#include "ie_parallel.hpp"
#include "mkldnn/omp_manager.h"

/* CPU "streams" implement a feature that allows multiple Infer Requests to be efficiently run simultaneously.
 * To avoid potential oversubscription the CPU execution resources are divided accordingly.
 * The feature enables much better performance for the networks that originally do not scale well with #threads
 * even for a large batches. Examples are lightweight topologies or topologies with many sequential/mem-bound/etc or
 * otherwise non-scalable layers. This is especially pronounced for many-core (e.g. server) machines.
 * This is rather throughput-oriented feature,because running multiple requests in parallel might increase the latency
 * of each request.
 * Additionally, the streams help to relax the need for the large batch to improve the throughput and simplify the
 * application logic, helping to saturate the CPU by multiple requests instead.
 * Implementation-wise, the "streams" constitute the following:
 *  - Pure "graph-less" Infer Requests that are not connected to the specific MKLDNNGraph (which is regular/legacy approach)
 *  - Just like regular requests, the graph-less go to the common (per ExecutableNetwork) queue
 *  - But unlike conventional case, there are multiple threads that grab the requests (see MultiWorkerTaskExecutor)
 *  - So every stream is in fact is independent "worker" thread that monitors the queue.
 *  - Every worker thread (stream) has it's own copy of the graph (which handles intermediate data required for execution)
 *  - While the Infer Requests just keep only input/output data
*/
namespace MKLDNNPlugin {

using namespace InferenceEngine;
class MKLDNNGraph;
class pinning_observer;

/* This structure handles an "execution context" - data required to execute an Infer Request.
 * This includes graph (which handles the intermediate data) and arena/observer for the TBB */
struct MultiWorkerTaskContext {
    std::shared_ptr<MKLDNNGraph> ptrGraph;
};

#if defined(__APPLE__) || defined(_WIN32)
typedef void cpu_set_t;
#define CPU_FREE(cpuset)
// notice that functions below are just stubs for OSs other than Linux
#endif
/* Check whether any affinity-related env variables are set (relevant for the OpenMP) */
bool check_env_variables();
/* Get the cores affinity mask for the current process */
bool get_process_mask(int& ncpus, cpu_set_t*& mask);
/* Pin current thread to a set of cores determined by the mask. */
bool pin_current_thread_by_mask(int ncores, const cpu_set_t* proc_mask);
/* Pin thread to a spare core in the round-robin scheme, while respecting the given process mask.
 * The function can also handle the hyper-threading (by populating the physical cores first) */
bool pin_thread_to_vacant_core(int thr_idx, int hyperthreads, int ncores, const cpu_set_t* proc_mask);

#if IE_THREAD == IE_THREAD_TBB
/* Simple observer that handles pinning threads to the cores, it serves as a callback for threads entering the arena. */
class pinning_observer: public tbb::task_scheduler_observer {
    cpu_set_t *mask;
    int ncpus;
    int stream_id, threads_per_stream;
    const int pinning_step;

public:
    pinning_observer(tbb::task_arena& _arena, int _stream_id, int _threads_per_stream, int _pinning_step = 1) :
            tbb::task_scheduler_observer(_arena),
            stream_id(_stream_id), threads_per_stream(_threads_per_stream), pinning_step(_pinning_step) {
        get_process_mask(ncpus, mask);
    }

    void on_scheduler_entry(bool) override {
        if (!mask) return;
        int thread_idx = tbb::task_arena::current_thread_index();
        int thr_idx = stream_id * threads_per_stream + thread_idx;
        // pin thread to the vacant slot
        pin_thread_to_vacant_core(thr_idx, pinning_step, ncpus, mask);
    }

    void on_scheduler_exit(bool) override {
        if (!mask) return;
        // reset the thread's mask (to the original process mask)
        pin_current_thread_by_mask(ncpus, mask);
    }

    ~pinning_observer() {
        if (mask)
            CPU_FREE(mask);
    }
};

class auto_scope_observing {
public:
     explicit auto_scope_observing(std::unique_ptr<tbb::task_scheduler_observer>&  _p) : p(_p) {
         if (p)
             p->observe(true);
     }
     ~auto_scope_observing() {
         if (p)
            p->observe(false);
     }

protected:
    std::unique_ptr<tbb::task_scheduler_observer>&  p;
};
#endif  // IE_THREAD == IE_THREAD_TBB

/* Class wrapping multiple worker threads that monitors the same queue with Infer Requests. */
class MultiWorkerTaskExecutor : public ITaskExecutor {
public:
    typedef std::shared_ptr<MultiWorkerTaskExecutor> Ptr;

    explicit MultiWorkerTaskExecutor(const std::vector<Task::Ptr>&, std::string name = "Default");

    ~MultiWorkerTaskExecutor();

    /**
    * @brief Adds task for execution and notifies one of the working threads about the new task.
    * @note can be called from multiple threads - tasks will be added to the queue and executed one-by-one in FIFO mode.
    * @param task - shared pointer to the task
    *  @return true if succeed to add task, otherwise - false
    */
    bool startTask(Task::Ptr task) override;

    static thread_local MultiWorkerTaskContext ptrContext;

private:
    std::vector<std::thread> _threads;
    std::mutex _queueMutex;
    std::condition_variable _queueCondVar;
    std::queue<Task::Ptr> _taskQueue;
    std::atomic<bool> _isStopped;
    std::string _name;
    std::atomic<int> _initCount;
};

/* Pure Infer Requests - just input and output data. */
class MKLDNNGraphlessInferRequest : public InferenceEngine::InferRequestInternal {
public:
    typedef std::shared_ptr<MKLDNNGraphlessInferRequest> Ptr;
    explicit MKLDNNGraphlessInferRequest(InferenceEngine::InputsDataMap networkInputs,
                                         InferenceEngine::OutputsDataMap networkOutputs);

    void InferImpl() override;

    void GetPerformanceCounts(std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> &perfMap) const override;

    /**
     * @brief Given optional implementation of setting blob to avoid need for it to be implemented by plugin
     * @param name - a name of input or output blob.
     * @param data - a reference to input or output blob. The type of Blob must correspond to the network input precision and size.
     */
    void SetBlob(const char *name, const InferenceEngine::Blob::Ptr &data) override;

    /**
     * @brief Given optional implementation of getting blob to avoid need for it to be implemented by plugin
     * @param name - a name of input or output blob.
     * @param data - a reference to input or output blob. The type of Blob must correspond to the network input precision and size.
     */
    void GetBlob(const char *name, InferenceEngine::Blob::Ptr &data) override;


    void SetBatch(int batch = -1) override;

private:
    int m_curBatch;
    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> m_perfMap;
};


}  // namespace MKLDNNPlugin
