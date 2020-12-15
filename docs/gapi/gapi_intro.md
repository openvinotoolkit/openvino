# Introduction to OpenCV Graph API (G-API)
OpenCV Graph API (G-API) is an OpenCV module targeted to make regular image and video processing fast and portable. G-API is a special module in OpenCV – in contrast with the majority of other main modules, this one acts as a framework rather than some specific CV algorithm. 

G-API is positioned as a next level optimization enabler for computer vision, focusing not on particular CV functions but on the whole algorithm optimization.

G-API provides means to define CV operations, construct graphs (in form of expressions) using it, and finally implement and run the operations for a particular backend.

The idea behind G-API is that if an algorithm can be expressed in a special embedded language (currently in C++), the framework can catch its sense and apply a number of optimizations to the whole thing automatically. Particular optimizations are selected based on which [kernels](https://docs.opencv.org/4.2.0/d0/d25/gapi_kernel_api.html) and [backends](https://docs.opencv.org/4.2.0/dc/d1c/group__gapi__std__backends.html) are involved in the graph compilation process, for example, the graph can be offloaded to GPU via the OpenCL backend, or optimized for memory consumption with the Fluid backend. Kernels, backends, and their settings are parameters to the graph compilation, so the graph itself does not depend on any platform-specific details and can be ported easily.

> **NOTE**: Graph API (G-API) was introduced in the most recent major OpenCV 4.0 release and now is being actively developed. The API is volatile at the moment and there may be minor but compatibility-breaking changes in the future.





