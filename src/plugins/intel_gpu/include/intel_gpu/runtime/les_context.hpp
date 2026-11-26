#pragma once

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>

namespace cldnn {

class stream;

class les_task_cancelled : public std::runtime_error {
public:
    les_task_cancelled() : std::runtime_error("LES task cancelled") {}
};

class les_context {
public:
    struct ticket {
        uint64_t epoch;
        std::shared_future<void> completion;
        uint64_t id;
    };

    explicit les_context(size_t queue_capacity, std::shared_ptr<stream> transfer_stream = {});
    ~les_context();
    les_context(const les_context&) = delete;
    les_context& operator=(const les_context&) = delete;

    uint64_t begin_epoch();
    ticket submit(uint64_t epoch, std::function<void()> work);
    std::optional<ticket> try_submit_speculative(uint64_t epoch, std::function<void()> work);
    bool promote(const ticket& pending);
    bool cancel_pending(const ticket& pending);
    void wait();
    void cancel();
    void stop();
    const std::shared_ptr<stream>& get_transfer_stream() const { return _transfer_stream; }

private:
    struct task {
        uint64_t epoch;
        bool speculative;
        uint64_t id;
        std::function<void()> work;
        std::promise<void> completion;
    };

    void run();
    std::optional<ticket> submit_task(uint64_t epoch, std::function<void()> work, bool speculative);
    void invalidate(bool stopping);
    void check_control_thread() const;
    static void fail_tasks(std::deque<task>& tasks, std::exception_ptr error);

    const size_t _capacity;
    std::shared_ptr<stream> _transfer_stream;
    std::mutex _mutex;
    std::mutex _control_mutex;
    std::condition_variable _changed;
    std::deque<task> _pending;
    uint64_t _epoch = 0;
    uint64_t _next_task_id = 0;
    bool _active = false;
    bool _invalidating = false;
    bool _stopping = false;
    std::exception_ptr _error;
    std::thread _worker;
    std::thread::id _worker_id;
};

}  // namespace cldnn