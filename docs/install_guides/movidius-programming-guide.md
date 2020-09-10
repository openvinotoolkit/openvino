# Intel® Movidius™ VPUs Programming Guide for Use with Intel® Distribution of OpenVINO™ toolkit {#openvino_docs_install_guides_movidius_programming_guide}

## See Also

- [Intel® Movidius™ VPUs Setup Guide for use with the Intel® Distribution of OpenVINO™](movidius-setup-guide.md)
- <a class="download" href="<domain_placeholder>/downloads/595850_Intel_Vision_Accelerator_Design_with_Intel_Movidius™_VPUs-HAL Configuration Guide_rev1.3.pdf">Intel® Vision Accelerator Design with Intel® Movidius™ VPUs HAL Configuration Guide</a>
- <a class="download" href="<domain_placeholder>/downloads/613514_Intel Vision Accelerator Design with Intel Movidius™ VPUs Workload Distribution_UG_r0.9.pdf">Intel® Vision Accelerator Design with Intel® Movidius™ VPUs Workload Distribution User Guide</a>
- <a class="download" href="<domain_placeholder>/downloads/613759_Intel Vision Accelerator Design with Intel Movidius™ VPUs Scheduler_UG_r0.9.pdf">Intel® Vision Accelerator Design with Intel® Movidius™ VPUs Scheduler User Guide</a>
- <a class="download" href="<domain_placeholder>/downloads/Intel Vision Accelerator Design with Intel Movidius™ VPUs Errata.pdf">Intel® Vision Accelerator Design with Intel® Movidius™ VPUs Errata</a>

The following section provides information on how to distribute a model across all 8 VPUs to maximize performance.

## Programming a C++ Application for the Accelerator

### Declare a Structure to Track Requests

The structure should hold:
1.	A pointer to an inference request.
2.	An ID to keep track of the request.
```cpp
struct Request {
    InferenceEngine::InferRequest::Ptr inferRequest;
    int frameidx;
};
```

### Declare a Vector of Requests

```cpp
// numRequests is the number of frames (max size, equal to the number of VPUs in use)
vector<Request> request(numRequests);
```

Declare and initialize 2 mutex variables:
1.	For each request
2.	For when all 8 requests are done

### Declare a Conditional Variable 

Conditional variable indicates when at most 8 requests are done at a time.

For inference requests, use the asynchronous IE API calls:

```cpp
// initialize infer request pointer – Consult IE API for more detail.
request[i].inferRequest = executable_network.CreateInferRequestPtr();
```

```cpp
// Run inference
request[i].inferRequest->StartAsync();
```


### Create a Lambda Function

Lambda Function enables the parsing and display of results.

Inside the Lambda body use the completion callback function:

```cpp
request[i].inferRequest->SetCompletionCallback
(nferenceEngine::IInferRequest::Ptr context)
```

## Additional Resources

- [Intel Distribution of OpenVINO Toolkit home page](https://software.intel.com/en-us/openvino-toolkit)

- [Intel Distribution of OpenVINO Toolkit documentation](https://docs.openvinotoolkit.org)
