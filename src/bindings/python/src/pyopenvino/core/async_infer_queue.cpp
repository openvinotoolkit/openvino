// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "pyopenvino/core/async_infer_queue.hpp"

#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

#include "pyopenvino/core/common.hpp"
#include "pyopenvino/core/infer_request.hpp"
#include "pyopenvino/utils/utils.hpp"

namespace py = pybind11;

class AsyncInferQueue {
public:
    AsyncInferQueue(ov::CompiledModel& model, size_t jobs) {
        if (jobs == 0) {
            jobs = static_cast<size_t>(Common::get_optimal_number_of_requests(model));
        }

        m_requests.reserve(jobs);
        m_user_ids.reserve(jobs);

        for (size_t handle = 0; handle < jobs; handle++) {
            // Create new "empty" InferRequestWrapper without pre-defined callback and
            // copy Inputs and Outputs from ov::CompiledModel
            m_requests.emplace_back(model.create_infer_request(), model.inputs(), model.outputs(), false);
            m_user_ids.push_back(py::none());
            m_idle_handles.push(handle);
        }

        this->set_default_callbacks();
    }

    ~AsyncInferQueue() {
        m_requests.clear();
    }

    bool _is_ready() {
        // Check if any request has finished already
        py::gil_scoped_release release;
        // acquire the mutex to access m_errors and m_idle_handles
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_errors.size() > 0)
            throw m_errors.front();
        return !(m_idle_handles.empty());
    }

    size_t get_idle_request_id() {
        // Wait for any request to complete and return its id
        // release GIL to avoid deadlock on python callback
        py::gil_scoped_release release;
        // acquire the mutex to access m_errors and m_idle_handles
        std::unique_lock<std::mutex> lock(m_mutex);
        m_cv.wait(lock, [this] {
            return !(m_idle_handles.empty());
        });
        size_t idle_handle = m_idle_handles.front();
        // wait for request to make sure it returned from callback
        m_requests[idle_handle].m_request->wait();
        if (m_errors.size() > 0)
            throw m_errors.front();
        return idle_handle;
    }

    void wait_all() {
        // Wait for all request to complete
        // release GIL to avoid deadlock on python callback
        py::gil_scoped_release release;
        for (auto&& request : m_requests) {
            request.m_request->wait();
        }
        // acquire the mutex to access m_errors
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_errors.size() > 0)
            throw m_errors.front();
    }

    void set_default_callbacks() {
        for (size_t handle = 0; handle < m_requests.size(); handle++) {
            // auto end_time = m_requests[handle].m_end_time; // TODO: pass it bellow? like in InferRequestWrapper

            m_requests[handle].m_request->set_callback([this, handle /* ... */](std::exception_ptr exception_ptr) {
                *m_requests[handle].m_end_time = Time::now();
                {
                    // acquire the mutex to access m_idle_handles
                    std::lock_guard<std::mutex> lock(m_mutex);
                    // Add idle handle to queue
                    m_idle_handles.push(handle);
                }
                // Notify locks in getIdleRequestId()
                m_cv.notify_one();

                try {
                    if (exception_ptr) {
                        std::rethrow_exception(exception_ptr);
                    }
                } catch (const std::exception& e) {
                    OPENVINO_THROW(e.what());
                }
            });
        }
    }

    void set_custom_callbacks(py::function f_callback) {
        // need to acquire GIL before py::function deletion
        auto callback_sp = Common::utils::wrap_pyfunction(std::move(f_callback));

        for (size_t handle = 0; handle < m_requests.size(); handle++) {
            m_requests[handle].m_request->set_callback([this, callback_sp, handle](std::exception_ptr exception_ptr) {
                *m_requests[handle].m_end_time = Time::now();
                if (exception_ptr == nullptr) {
                    // Acquire GIL, execute Python function
                    py::gil_scoped_acquire acquire;
                    try {
                        (*callback_sp)(m_requests[handle], m_user_ids[handle]);
                    } catch (const py::error_already_set& py_error) {
                        // This should behave the same as assert(!PyErr_Occurred())
                        // since constructor for pybind11's error_already_set is
                        // performing PyErr_Fetch which clears error indicator and
                        // saves it inside itself.
                        assert(py_error.type());
                        // acquire the mutex to access m_errors
                        std::lock_guard<std::mutex> lock(m_mutex);
                        m_errors.push(py_error);
                    }
                }

                {
                    // acquire the mutex to access m_idle_handles
                    std::lock_guard<std::mutex> lock(m_mutex);
                    // Add idle handle to queue
                    m_idle_handles.push(handle);
                }
                // Notify locks in getIdleRequestId()
                m_cv.notify_one();

                try {
                    if (exception_ptr) {
                        std::rethrow_exception(exception_ptr);
                    }
                } catch (const std::exception& e) {
                    OPENVINO_THROW(e.what());
                }
            });
        }
    }

    // AsyncInferQueue is the owner of all requests. When AsyncInferQueue is destroyed,
    // all of requests are destroyed as well.
    std::vector<InferRequestWrapper> m_requests;
    std::queue<size_t> m_idle_handles;
    std::vector<py::object> m_user_ids;  // user ID can be any Python object
    std::mutex m_mutex;
    std::condition_variable m_cv;
    std::queue<py::error_already_set> m_errors;
};

void regclass_AsyncInferQueue(py::module m) {
    py::class_<AsyncInferQueue, std::shared_ptr<AsyncInferQueue>> cls(m, "AsyncInferQueue");
    cls.doc() = "openvino.AsyncInferQueue represents a helper that creates a pool of asynchronous"
                "InferRequests and provides synchronization functions to control flow of a simple pipeline.";

    cls.def(py::init<ov::CompiledModel&, size_t>(),
            py::arg("model"),
            py::arg("jobs") = 0,
            R"(
                Creates AsyncInferQueue.

                :param model: Model to be used to create InferRequests in a pool.
                :type model: openvino.CompiledModel
                :param jobs: Number of InferRequests objects in a pool. If 0, jobs number
                will be set automatically to the optimal number. Default: 0
                :type jobs: int
                :rtype: openvino.AsyncInferQueue
            )");

    // Overload for single input, it will throw error if a model has more than one input.
    cls.def(
        "start_async",
        [](AsyncInferQueue& self, const ov::Tensor& inputs, py::object userdata) {
            // getIdleRequestId function has an intention to block InferQueue
            // until there is at least one idle (free to use) InferRequest
            auto handle = self.get_idle_request_id();
            {
                std::lock_guard<std::mutex> lock(self.m_mutex);
                self.m_idle_handles.pop();
            }
            // Set new inputs label/id from user
            self.m_user_ids[handle] = userdata;
            // Update inputs if there are any
            self.m_requests[handle].m_request->set_input_tensor(inputs);
            // Now GIL can be released - we are NOT working with Python objects in this block
            {
                py::gil_scoped_release release;
                *self.m_requests[handle].m_start_time = Time::now();
                // Start InferRequest in asynchronus mode
                self.m_requests[handle].m_request->start_async();
            }
        },
        py::arg("inputs"),
        py::arg("userdata"),
        R"(
            Run asynchronous inference using the next available InferRequest.

            This function releases the GIL, so another Python thread can
            work while this function runs in the background.

            :param inputs: Data to set on single input tensor of next available InferRequest from
            AsyncInferQueue's pool.
            :type inputs: openvino.Tensor
            :param userdata: Any data that will be passed to a callback
            :type userdata: Any
            :rtype: None

            GIL is released while waiting for the next available InferRequest.
        )");

    // Overload for general case, it accepts dict of inputs that are pairs of (key, value).
    // Where keys types are:
    // * ov::Output<const ov::Node>
    // * py::str (std::string)
    // * py::int_ (size_t)
    // and values are always of type: ov::Tensor.
    cls.def(
        "start_async",
        [](AsyncInferQueue& self, const py::dict& inputs, py::object userdata) {
            // getIdleRequestId function has an intention to block InferQueue
            // until there is at least one idle (free to use) InferRequest
            auto handle = self.get_idle_request_id();
            {
                std::lock_guard<std::mutex> lock(self.m_mutex);
                self.m_idle_handles.pop();
            }
            // Set new inputs label/id from user
            self.m_user_ids[handle] = userdata;
            // Update inputs if there are any
            Common::set_request_tensors(*self.m_requests[handle].m_request, inputs);
            // Now GIL can be released - we are NOT working with Python objects in this block
            {
                py::gil_scoped_release release;
                *self.m_requests[handle].m_start_time = Time::now();
                // Start InferRequest in asynchronus mode
                self.m_requests[handle].m_request->start_async();
            }
        },
        py::arg("inputs"),
        py::arg("userdata"),
        R"(
            Run asynchronous inference using the next available InferRequest.

            This function releases the GIL, so another Python thread can
            work while this function runs in the background.

            :param inputs: Data to set on input tensors of next available InferRequest from
            AsyncInferQueue's pool.
            :type inputs: dict[Union[int, str, openvino.ConstOutput] : openvino.Tensor]
            :param userdata: Any data that will be passed to a callback
            :rtype: None

            GIL is released while waiting for the next available InferRequest.
        )");

    cls.def("is_ready",
            &AsyncInferQueue::_is_ready,
            R"(
            One of 'flow control' functions.
            Returns True if any free request in the pool, otherwise False.

            GIL is released while running this function.

            :return: If there is at least one free InferRequest in a pool, returns True.
            :rtype: bool
    )");

    cls.def("wait_all",
            &AsyncInferQueue::wait_all,
            R"(
            One of 'flow control' functions. Blocking call.
            Waits for all InferRequests in a pool to finish scheduled work.

            GIL is released while running this function.
        )");

    cls.def("get_idle_request_id",
            &AsyncInferQueue::get_idle_request_id,
            R"(
            Returns next free id of InferRequest from queue's pool.
            Function waits for any request to complete and then returns this request's id.

            GIL is released while running this function.

            :rtype: int
        )");

    cls.def("set_callback",
            &AsyncInferQueue::set_custom_callbacks,
            R"(
            Sets unified callback on all InferRequests from queue's pool.
            Signature of such function should have two arguments, where
            first one is InferRequest object and second one is userdata
            connected to InferRequest from the AsyncInferQueue's pool.

            .. code-block:: python

                def f(request, userdata):
                    result = request.output_tensors[0]
                    print(result + userdata)

                async_infer_queue.set_callback(f)

            :param callback: Any Python defined function that matches callback's requirements.
            :type callback: function
        )");

    cls.def(
        "__len__",
        [](AsyncInferQueue& self) {
            return self.m_requests.size();
        },
        R"(
        Number of InferRequests in the pool.
        
        :rtype: int
    )");

    cls.def(
        "__iter__",
        [](AsyncInferQueue& self) {
            return py::make_iterator(self.m_requests.begin(), self.m_requests.end());
        },
        py::keep_alive<0, 1>()); /* Keep set alive while iterator is used */

    cls.def(
        "__getitem__",
        [](AsyncInferQueue& self, size_t i) {
            return self.m_requests[i];
        },
        R"(
        :param i: InferRequest id
        :type i: int
        :return: InferRequests from the pool with given id.
        :rtype: openvino.InferRequest
    )");

    cls.def_property_readonly(
        "userdata",
        [](AsyncInferQueue& self) {
            return self.m_user_ids;
        },
        R"(
        :return: List of all passed userdata. List is filled with `None` if the data wasn't passed yet.
        :rtype: List[Any]
    )");

    cls.def("__repr__", [](const AsyncInferQueue& self) {
        return "<" + Common::get_class_name(self) + ": " + std::to_string(self.m_requests.size()) + " jobs>";
    });
}
