# Building a Face Analytics Pipeline {#openvino_docs_gapi_gapi_face_analytics_pipeline}

## Overview
In this tutorial you will learn:

* How to integrate Deep Learning inference in a G-API graph.
* How to run a G-API graph on a video stream and obtain data from it.

## Prerequisites
This sample requires:

* PC with GNU/Linux or Microsoft Windows (Apple macOS is supported but was not tested)
* OpenCV 4.2 or higher built with [Intel® Distribution of OpenVINO™ Toolkit](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html) (building with [Intel® TBB](https://www.threadingbuildingblocks.org/intel-tbb-tutorial) is a plus)
* The following pre-trained models from the [Open Model Zoo](@ref omz_models_group_intel):
    * [face-detection-adas-0001](@ref omz_models_model_face_detection_adas_0001)
    * [age-gender-recognition-retail-0013](@ref omz_models_model_age_gender_recognition_retail_0013)
    * [emotions-recognition-retail-0003](@ref omz_models_model_emotions_recognition_retail_0003)

To download the models from the Open Model Zoo, use the [Model Downloader](@ref omz_tools_downloader) tool.

## Introduction: Why G-API
Many computer vision algorithms run on a video stream rather than on individual images. Stream processing usually consists of multiple steps – like decode, preprocessing, detection, tracking, classification (on detected objects), and visualization – forming a *video processing pipeline*. Moreover, many these steps of such pipeline can run in parallel – modern platforms have different hardware blocks on the same chip like decoders and GPUs, and extra accelerators can be plugged in as extensions for deep learning offload.

Given all this manifold of options and a variety in video analytics algorithms, managing such pipelines effectively quickly becomes a problem. For sure it can be done manually, but this approach doesn't scale: if a change is required in the algorithm (e.g. a new pipeline step is added), or if it is ported on a new platform with different capabilities, the whole pipeline needs to be re-optimized.

Starting with version 4.2, OpenCV offers a solution to this problem. OpenCV G-API now can manage Deep Learning inference (a cornerstone of any modern analytics pipeline) with a traditional Computer Vision as well as video capturing/decoding, all in a single pipeline. G-API takes care of pipelining itself – so if the algorithm or platform changes, the execution model adapts to it automatically.

## Pipeline Overview
Our sample application is based on [Interactive Face Detection](@ref omz_demos_interactive_face_detection_demo_cpp) demo from Open Model Zoo. A simplified pipeline consists of the following steps:

1. Image acquisition and decode
2. Detection with preprocessing
3. Classification with preprocessing for every detected object with two networks
4. Visualization

![Face Analytics Pipeline Overview](../img/gapi_face_analytics_pipeline.png)

## Construct a pipeline {#gapi_ifd_constructing}

Constructing a G-API graph for a video streaming case does not differ much from a [regular usage](https://docs.opencv.org/4.5.0/d0/d1e/gapi.html#gapi_example) of G-API -- it is still about defining graph *data* (with cv::GMat, `cv::GScalar`, and `cv::GArray`) and *operations* over it. Inference also becomes an operation in the graph, but is defined in a little bit different way.

### Declare Deep Learning topologies {#gapi_ifd_declaring_nets}

In contrast with traditional CV functions (see [core](https://docs.opencv.org/4.5.0/df/d1f/group__gapi__core.html) and [imgproc](https://docs.opencv.org/4.5.0/d2/d00/group__gapi__imgproc.html)) where G-API declares distinct operations for every function, inference in G-API is a single generic operation `cv::gapi::infer<>`. As usual, it is just an interface and it can be implemented in a number of ways under the hood. In OpenCV 4.2, only OpenVINO™ Runtime-based backend is available, and OpenCV's own DNN module-based backend is to come.

`cv::gapi::infer<>` is _parametrized_ by the details of a topology we are going to execute. Like operations, topologies in G-API are strongly typed and are defined with a special macro `G_API_NET()`:

```cpp
// Face detector: takes one Mat, returns another Mat
G_API_NET(Faces, <cv::GMat(cv::GMat)>, "face-detector");
// Age/Gender recognition - takes one Mat, returns two:
// one for Age and one for Gender. In G-API, multiple-return-value operations
// are defined using std::tuple<>.
using AGInfo = std::tuple<cv::GMat, cv::GMat>;
G_API_NET(AgeGender, <AGInfo(cv::GMat)>,   "age-gender-recoginition");
// Emotion recognition - takes one Mat, returns another.
G_API_NET(Emotions, <cv::GMat(cv::GMat)>, "emotions-recognition");
```

Similar to how operations are defined with `G_API_OP()`, network description requires three parameters:
1. A type name. Every defined topology is declared as a distinct C++ type which is used further in the program -- see below.
2. A `std::function<>`-like API signature. G-API traits networks as regular "functions" which take and return data. Here network `Faces` (a detector) takes a `cv::GMat` and returns a `cv::GMat`, while network `AgeGender` is known to provide two outputs (age and gender blobs, respectively) -- so its has a `std::tuple<>` as a return type.
3. A topology name -- can be any non-empty string, G-API is using these names to distinguish networks inside. Names should be unique in the scope of a single graph.

## Building a GComputation {#gapi_ifd_gcomputation}

Now the above pipeline is expressed in G-API like this:

```cpp
cv::GComputation pp([]() {
    // Declare an empty GMat - the beginning of the pipeline.
    cv::GMat in;
    // Run face detection on the input frame. Result is a single GMat,
    // internally representing an 1x1x200x7 SSD output.
    // This is a single-patch version of infer:
    // - Inference is running on the whole input image;
    // - Image is converted and resized to the network's expected format
    //   automatically.
    cv::GMat detections = cv::gapi::infer<custom::Faces>(in);
    // Parse SSD output to a list of ROI (rectangles) using
    // a custom kernel. Note: parsing SSD may become a "standard" kernel.
    cv::GArray<cv::Rect> faces = custom::PostProc::on(detections, in);
    // Now run Age/Gender model on every detected face. This model has two
    // outputs (for age and gender respectively).
    // A special ROI-list-oriented form of infer<>() is used here:
    // - First input argument is the list of rectangles to process,
    // - Second one is the image where to take ROI from;
    // - Crop/Resize/Layout conversion happens automatically for every image patch
    //   from the list
    // - Inference results are also returned in form of list (GArray<>)
    // - Since there're two outputs, infer<> return two arrays (via std::tuple).
    cv::GArray<cv::GMat> ages;
    cv::GArray<cv::GMat> genders;
    std::tie(ages, genders) = cv::gapi::infer<custom::AgeGender>(faces, in);
    // Recognize emotions on every face.
    // ROI-list-oriented infer<>() is used here as well.
    // Since custom::Emotions network produce a single output, only one
    // GArray<> is returned here.
    cv::GArray<cv::GMat> emotions = cv::gapi::infer<custom::Emotions>(faces, in);
    // Return the decoded frame as a result as well.
    // Input matrix can't be specified as output one, so use copy() here
    // (this copy will be optimized out in the future).
    cv::GMat frame = cv::gapi::copy(in);
    // Now specify the computation's boundaries - our pipeline consumes
    // one images and produces five outputs.
    return cv::GComputation(cv::GIn(in),
                            cv::GOut(frame, faces, ages, genders, emotions));
});
```

Every pipeline starts with declaring empty data objects – which act as inputs to the pipeline. Then we call a generic `cv::gapi::infer<>` specialized to Faces detection network. `cv::gapi::infer<>` inherits its signature from its template parameter – and in this case it expects one input cv::GMat and produces one output cv::GMat.

In this sample we use a pre-trained SSD-based network and its output needs to be parsed to an array of detections (object regions of interest, ROIs). It is done by a custom operation custom::PostProc, which returns an array of rectangles (of type `cv::GArray<cv::Rect>`) back to the pipeline. This operation also filters out results by a confidence threshold – and these details are hidden in the kernel itself. Still, at the moment of graph construction we operate with interfaces only and don't need actual kernels to express the pipeline – so the implementation of this post-processing will be listed later.

After detection result output is parsed to an array of objects, we can run classification on any of those. G-API doesn't support syntax for in-graph loops like `for_each()` yet, but instead `cv::gapi::infer<>` comes with a special list-oriented overload.

User can call `cv::gapi::infer<>` with a `cv::GArray` as the first argument, so then G-API assumes it needs to run the associated network on every rectangle from the given list of the given frame (second argument). Result of such operation is also a list – a cv::GArray of `cv::GMat`.

Since AgeGender network itself produces two outputs, it's output type for a list-based version of `cv::gapi::infer` is a tuple of arrays. We use `std::tie()` to decompose this input into two distinct objects.

Emotions network produces a single output so its list-based inference's return type is `cv::GArray<cv::GMat>`.

## Configure the Pipeline {#gapi_ifd_configuration}

G-API strictly separates construction from configuration -- with the idea to keep algorithm code itself platform-neutral. In the above listings we only declared our operations and expressed the overall data flow, but didn't even mention that we use OpenVINO™. We only described *what* we do, but not *how* we do it. Keeping these two aspects clearly separated is the design goal for G-API.

Platform-specific details arise when the pipeline is *compiled* -- i.e. is turned from a declarative to an executable form. The way *how* to run stuff is specified via compilation arguments, and new inference/streaming features are no exception from this rule. 

G-API is built on backends which implement interfaces (see [Architecture](https://docs.opencv.org/4.5.0/de/d4d/gapi_hld.html) and [Kernels](kernel_api.md) for details) -- thus `cv::gapi::infer<>` is a function which can be implemented by different backends. In OpenCV 4.2, only OpenVINO™ Runtime backend for inference is available. Every inference backend in G-API has to provide a special parameterizable structure to express *backend-specific* neural network parameters -- and in this case, it is `cv::gapi::ie::Params`:

```cpp
auto det_net = cv::gapi::ie::Params<custom::Faces> {
    cmd.get<std::string>("fdm"),   // read cmd args: path to topology IR
    cmd.get<std::string>("fdw"),   // read cmd args: path to weights
    cmd.get<std::string>("fdd"),   // read cmd args: device specifier
};
auto age_net = cv::gapi::ie::Params<custom::AgeGender> {
    cmd.get<std::string>("agem"),   // read cmd args: path to topology IR
    cmd.get<std::string>("agew"),   // read cmd args: path to weights
    cmd.get<std::string>("aged"),   // read cmd args: device specifier
}.cfgOutputLayers({ "age_conv3", "prob" });
auto emo_net = cv::gapi::ie::Params<custom::Emotions> {
    cmd.get<std::string>("emom"),   // read cmd args: path to topology IR
    cmd.get<std::string>("emow"),   // read cmd args: path to weights
    cmd.get<std::string>("emod"),   // read cmd args: device specifier
};
```

Here we define three parameter objects: `det_net`, `age_net`, and `emo_net`. Every object is a `cv::gapi::ie::Params` structure parametrization for each particular network we use. On a compilation stage, G-API automatically matches network parameters with their `cv::gapi::infer<>` calls in graph using this information.

Regardless of the topology, every parameter structure is constructed with three string arguments – specific to the OpenVINO™ Runtime:

* Path to the topology's intermediate representation (.xml file);
* Path to the topology's model weights (.bin file);
* Device where to run – "CPU", "GPU", and others – based on your OpenVINO™ Toolkit installation. These arguments are taken from the command-line parser.

Once networks are defined and custom kernels are implemented, the pipeline is compiled for streaming:
```cpp
// Form a kernel package (with a single OpenCV-based implementation of our
// post-processing) and a network package (holding our three networks).
auto kernels = cv::gapi::kernels<custom::OCVPostProc>();
auto networks = cv::gapi::networks(det_net, age_net, emo_net);
// Compile our pipeline and pass our kernels & networks as
// parameters.  This is the place where G-API learns which
// networks & kernels we're actually operating with (the graph
// description itself known nothing about that).
auto cc = pp.compileStreaming(cv::compile_args(kernels, networks));
```

`cv::GComputation::compileStreaming()` triggers a special video-oriented form of graph compilation where G-API is trying to optimize throughput. Result of this compilation is an object of special type `cv::GStreamingCompiled` – in contrast to a traditional callable `cv::GCompiled`, these objects are closer to media players in their semantics.

> **NOTE**: There is no need to pass metadata arguments describing the format of the input video stream in `cv::GComputation::compileStreaming()` – G-API figures automatically what are the formats of the input vector and adjusts the pipeline to these formats on-the-fly. User still can pass metadata there as with regular `cv::GComputation::compile()` in order to fix the pipeline to the specific input format.

## Running the Pipeline  {#gapi_ifd_running}

Pipelining optimization is based on processing multiple input video frames simultaneously, running different steps of the pipeline in parallel. This is why it works best when the framework takes full control over the video stream.

The idea behind streaming API is that user specifies an *input source* to the pipeline and then G-API manages its execution automatically until the source ends or user interrupts the execution. G-API pulls new image data from the source and passes it to the pipeline for processing.

Streaming sources are represented by the interface `cv::gapi::wip::IStreamSource`. Objects implementing this interface may be passed to `GStreamingCompiled` as regular inputs via `cv::gin()` helper function. In OpenCV 4.2, only one streaming source is allowed per pipeline -- this requirement will be relaxed in the future.

OpenCV comes with a great class cv::VideoCapture and by default G-API ships with a stream source class based on it -- `cv::gapi::wip::GCaptureSource`. Users can implement their own
streaming sources e.g. using [VAAPI](https://01.org/vaapi) or other Media or Networking APIs.

Sample application specifies the input source as follows:
```cpp
auto in_src = cv::gapi::wip::make_src<cv::gapi::wip::GCaptureSource>(input);
cc.setSource(cv::gin(in_src));
```

Please note that a GComputation may still have multiple inputs like `cv::GMat`, `cv::GScalar`, or `cv::GArray` objects. User can pass their respective host-side types (`cv::Mat`, `cv::Scalar`, `std::vector<>`) in the input vector as well, but in Streaming mode these objects will create "endless" constant streams. Mixing a real video source stream and a const data stream is allowed.

Running a pipeline is easy – just call `cv::GStreamingCompiled::start()` and fetch your data with blocking `cv::GStreamingCompiled::pull()` or non-blocking `cv::GStreamingCompiled::try_pull()`; repeat until the stream ends:

```cpp
// After data source is specified, start the execution
cc.start();
// Declare data objects we will be receiving from the pipeline.
cv::Mat frame;                      // The captured frame itself
std::vector<cv::Rect> faces;        // Array of detected faces
std::vector<cv::Mat> out_ages;      // Array of inferred ages (one blob per face)
std::vector<cv::Mat> out_genders;   // Array of inferred genders (one blob per face)
std::vector<cv::Mat> out_emotions;  // Array of classified emotions (one blob per face)
// Implement different execution policies depending on the display option
// for the best performance.
while (cc.running()) {
    auto out_vector = cv::gout(frame, faces, out_ages, out_genders, out_emotions);
    if (no_show) {
        // This is purely a video processing. No need to balance
        // with UI rendering.  Use a blocking pull() to obtain
        // data. Break the loop if the stream is over.
        if (!cc.pull(std::move(out_vector)))
            break;
    } else if (!cc.try_pull(std::move(out_vector))) {
        // Use a non-blocking try_pull() to obtain data.
        // If there's no data, let UI refresh (and handle keypress)
        if (cv::waitKey(1) >= 0) break;
        else continue;
    }
    // At this point we have data for sure (obtained in either
    // blocking or non-blocking way).
    frames++;
    labels::DrawResults(frame, faces, out_ages, out_genders, out_emotions);
    labels::DrawFPS(frame, frames, avg.fps(frames));
    if (!no_show) cv::imshow("Out", frame);
}
```

The above code may look complex but in fact it handles two modes – with and without graphical user interface (GUI):

* When a sample is running in a "headless" mode (`--pure` option is set), this code simply pulls data from the pipeline with the blocking `pull()` until it ends. This is the most performant mode of execution.
* When results are also displayed on the screen, the Window System needs to take some time to refresh the window contents and handle GUI events. In this case, the demo pulls data with a non-blocking `try_pull()` until there is no more data available (but it does not mark end of the stream – just means new data is not ready yet), and only then displays the latest obtained result and refreshes the screen. Reducing the time spent in GUI with this trick increases the overall performance a little bit.

## Comparison with Serial Mode
The sample can also run in a serial mode for a reference and benchmarking purposes. In this case, a regular `cv::GComputation::compile()` is used and a regular single-frame `cv::GCompiled` object is produced; the pipelining optimization is not applied within G-API; it is the user responsibility to acquire image frames from `cv::VideoCapture` object and pass those to G-API.

```cpp
cv::VideoCapture cap(input);
cv::Mat in_frame, frame;            // The captured frame itself
std::vector<cv::Rect> faces;        // Array of detected faces
std::vector<cv::Mat> out_ages;      // Array of inferred ages (one blob per face)
std::vector<cv::Mat> out_genders;   // Array of inferred genders (one blob per face)
std::vector<cv::Mat> out_emotions;  // Array of classified emotions (one blob per face)
while (cap.read(in_frame)) {
    pp.apply(cv::gin(in_frame),
             cv::gout(frame, faces, out_ages, out_genders, out_emotions),
             cv::compile_args(kernels, networks));
    labels::DrawResults(frame, faces, out_ages, out_genders, out_emotions);
    frames++;
    if (frames == 1u) {
        // Start timer only after 1st frame processed -- compilation
        // happens on-the-fly here
        avg.start();
    } else {
        // Measurfe & draw FPS for all other frames
        labels::DrawFPS(frame, frames, avg.fps(frames-1));
    }
    if (!no_show) {
        cv::imshow("Out", frame);
        if (cv::waitKey(1) >= 0) break;
    }
}
```

On a test machine (Intel® Core™ i5-6600), with OpenCV built with [Intel® TBB](https://www.threadingbuildingblocks.org/intel-tbb-tutorial) support, detector network assigned to CPU, and classifiers to iGPU, the pipelined sample outperformes the serial one by the factor of 1.36x (thus adding +36% in overall throughput).

## Conclusion
G-API introduces a technological way to build and optimize hybrid pipelines. Switching to a new execution model does not require changes in the algorithm code expressed with G-API – only the way how graph is triggered differs.

## Listing: Post-Processing Kernel
G-API gives an easy way to plug custom code into the pipeline even if it is running in a streaming mode and processing tensor data. Inference results are represented by multi-dimensional `cv::Mat` objects so accessing those is as easy as with a regular DNN module.

The OpenCV-based SSD post-processing kernel is defined and implemented in this sample as follows:
```cpp
// SSD Post-processing function - this is not a network but a kernel.
// The kernel body is declared separately, this is just an interface.
// This operation takes two Mats (detections and the source image),
// and returns a vector of ROI (filtered by a default threshold).
// Threshold (or a class to select) may become a parameter, but since
// this kernel is custom, it doesn't make a lot of sense.
G_API_OP(PostProc, <cv::GArray<cv::Rect>(cv::GMat, cv::GMat)>, "custom.fd_postproc") {
    static cv::GArrayDesc outMeta(const cv::GMatDesc &, const cv::GMatDesc &) {
        // This function is required for G-API engine to figure out
        // what the output format is, given the input parameters.
        // Since the output is an array (with a specific type),
        // there's nothing to describe.
        return cv::empty_array_desc();
    }
};
// OpenCV-based implementation of the above kernel.
GAPI_OCV_KERNEL(OCVPostProc, PostProc) {
    static void run(const cv::Mat &in_ssd_result,
                    const cv::Mat &in_frame,
                    std::vector<cv::Rect> &out_faces) {
        const int MAX_PROPOSALS = 200;
        const int OBJECT_SIZE   =   7;
        const cv::Size upscale = in_frame.size();
        const cv::Rect surface({0,0}, upscale);
        out_faces.clear();
        const float *data = in_ssd_result.ptr<float>();
        for (int i = 0; i < MAX_PROPOSALS; i++) {
            const float image_id   = data[i * OBJECT_SIZE + 0]; // batch id
            const float confidence = data[i * OBJECT_SIZE + 2];
            const float rc_left    = data[i * OBJECT_SIZE + 3];
            const float rc_top     = data[i * OBJECT_SIZE + 4];
            const float rc_right   = data[i * OBJECT_SIZE + 5];
            const float rc_bottom  = data[i * OBJECT_SIZE + 6];
            if (image_id < 0.f) {  // indicates end of detections
                break;
            }
            if (confidence < 0.5f) { // a hard-coded snapshot
                continue;
            }
            // Convert floating-point coordinates to the absolute image
            // frame coordinates; clip by the source image boundaries.
            cv::Rect rc;
            rc.x      = static_cast<int>(rc_left   * upscale.width);
            rc.y      = static_cast<int>(rc_top    * upscale.height);
            rc.width  = static_cast<int>(rc_right  * upscale.width)  - rc.x;
            rc.height = static_cast<int>(rc_bottom * upscale.height) - rc.y;
            out_faces.push_back(rc & surface);
        }
    }
};
```