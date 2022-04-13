# Graph API Kernel API {#openvino_docs_gapi_kernel_api}

Main idea behind Graph API (G-API) is mobility – a pipeline built with G-API must, at least be able to, be portable. It means that either it works out-of-the box when compiled for new platform, or G-API provides necessary tools to make it running there, with little-to-no changes in the algorithm itself.

This can be achieved by separating kernel interface from its implementation. Once pipeline is built, using kernel interfaces, it becomes implementation-neutral. The implementation details (i.e. which kernels to use) are passed on in a separate stage - graph compilation.

Examples of kernel-implementation hierarchy:
![Kernel API/implementation hierarchy example](../img/gapi_kernel_implementation_hierarchy.png)

Pipeline itself then can only be expressed in terms of `A`, `B`, and so on, and choosing which implementation to use in execution becomes an external parameter.

## Define a Kernel
G-API provides a macro to define a new kernel interface - `G_TYPED_KERNEL()`:

```cpp
#include <opencv2/gapi.hpp>
G_TYPED_KERNEL(GFilter2D,
               <cv::GMat(cv::GMat,int,cv::Mat,cv::Point,double,int,cv::Scalar)>,
               "org.opencv.imgproc.filters.filter2D")
{
    static cv::GMatDesc                 // outMeta's return value type
    outMeta(cv::GMatDesc    in       ,  // descriptor of input GMat
            int             ddepth   ,  // depth parameter
            cv::Mat      /* coeffs */,  // (unused)
            cv::Point    /* anchor */,  // (unused)
            double       /* scale  */,  // (unused)
            int          /* border */,  // (unused)
            cv::Scalar   /* bvalue */ ) // (unused)
    {
        return in.withDepth(ddepth);
    }
};
```

This macro is a shortcut to a new type definition. It takes three arguments to register a new type, and requires type body to be present (see below). The macro arguments are:

* Kernel interface name -- Also serves as a name of new type defined with this macro;
* Kernel signature -- An `std::function<>`-like signature which defines API of the kernel;
* Kernel unique name -- Used to identify kernel when its type information is stripped within the system;
* Kernel declaration/Function declaration -- In both cases, a new entity must be used then according to the way it was defined.

Kernel signature defines kernel's usage syntax, which parameters it takes during graph construction. Implementations can also use this signature to derive it into backend-specific callback signatures (check next chapter).

Kernel may accept values of any type, and G-API dynamic types are handled in a special way. All other types are unclear to G-API, and are passed to kernel in `outMeta()` or in execution callbacks as-is.

Kernel return value can only take form of G-API dynamic type -– `cv::GMat`, `cv::GScalar`, or `cv::GArray<T>`. If an operation has more than one output, it should be wrapped into an `std::tuple<>` (which can contain only previously mentioned G-API types). Arbitrary-output-number operations are not supported.

Once a kernel is defined, it can be used in pipelines with special G-API-supplied method -- `on()`. This method has the same signature as defined in kernel, so the following code is a properly constructed:

```cpp
cv::GMat in;
cv::GMat out = GFilter2D::on(/* GMat    */  in,
                             /* int     */  -1,
                             /* Mat     */  conv_kernel_mat,
                             /* Point   */  cv::Point(-1,-1),
                             /* double  */  0.,
                             /* int     */  cv::BORDER_DEFAULT,
                             /* Scalar  */  cv::Scalar(0));
```
This example has some verbosity. Usually a kernel declaration comes with a C++ function wrapper (*"factory method"*), which enables optional parameters, more compact syntax, Doxygen comments, etc.:

```cpp
cv::GMat filter2D(cv::GMat   in,
                  int        ddepth,
                  cv::Mat    k,
                  cv::Point  anchor  = cv::Point(-1,-1),
                  double     scale   = 0.,
                  int        border  = cv::BORDER_DEFAULT,
                  cv::Scalar bval    = cv::Scalar(0))
{
    return GFilter2D::on(in, ddepth, k, anchor, scale, border, bval);
}
```
So now it can be used like:
```cpp
cv::GMat in;
cv::GMat out = filter2D(in, -1, conv_kernel_mat);
```

### Extra information
In the current version, kernel declaration body (everything within the curly braces) must contain a static function `outMeta()`. This function establish a functional dependency between operation input and output metadata.

Metadata are informations about data, on which kernel operates on. Since non-G-API types are unclear to G-API, G-API cares only about G* data descriptors (i.e. dimensions and format of `cv::GMat`, etc).

`outMeta()` is also an example of how kernel signature can be transformed into a derived callback. In this example, `outMeta()` signature exactly follows the kernel signature (defined within the macro) but is different. Where kernel expects `cv::GMat`, `outMeta()` takes and returns `cv::GMatDesc` (a G-API structure metadata for `cv::GMat`).

The purpouse of `outMeta()` is to spread metadata information within computation from inputs to outputs and infer metadata of internal (intermediate, temporary) data objects. This information is required for further pipeline optimizations, memory allocation, and other operations done by G-API framework during graph compilation.

## Implement a Kernel
Once a kernel is declared, its interface can be used to implement its versions in different backends. This concept is taken from object-oriented programming "Interface/Implementation" idiom: an interface can be implemented multiple times, and different implementations of a kernel should be substitutable with each other without breaking the algorithm (pipeline) logic (Liskov Substitution Principle).

As standard every backend defines its own way to implement a kernel interface. Although – dependable on which plugin is used, its kernel implementation must be "derived" from a kernel interface type.

Kernels implementation are then organized into kernel packages. Those packages are passed to `cv::GComputation::compile()` as compile arguments, with some hints to G-API on how to select proper kernels (To learn more, check "Heterogeneity"[TBD]).

For example, the aforementioned Filter2D is implemented in "reference" CPU (OpenCV) plugin this way: 

> **NOTE:** Below is a simplified form with improper border handling:

```cpp
#include <opencv2/gapi/cpu/gcpukernel.hpp>     // GAPI_OCV_KERNEL()
#include <opencv2/imgproc.hpp>                 // cv::filter2D()
GAPI_OCV_KERNEL(GCPUFilter2D, GFilter2D)
{
    static void
    run(const cv::Mat    &in,       // in - derived from GMat
        const int         ddepth,   // opaque (passed as-is)
        const cv::Mat    &k,        // opaque (passed as-is)
        const cv::Point  &anchor,   // opaque (passed as-is)
        const double      delta,    // opaque (passed as-is)
        const int         border,   // opaque (passed as-is)
        const cv::Scalar &,         // opaque (passed as-is)
        cv::Mat          &out)      // out - derived from GMat (retval)
    {
        cv::filter2D(in, out, ddepth, k, anchor, delta, border);
    }
};
```
Notice how CPU (OpenCV) plugin has transformed the original kernel signature:

* Input `cv::GMat` has been substituted with `cv::Mat`, holding actual input data for the underlying OpenCV function call;
* Output `cv::GMat `has been transformed into extra output parameter, thus `GCPUFilter2D::run()` takes one argument more than the original kernel signature.

Kernel developer shouldn't interest where `cv::Mat` objects come from, instead of the original `cv::GMat`, and should just follow the signature conventions defined by the plugin. G-API will call this method during execution and supply all the necessary information (and forward the original opaque data as-is).

## Compound Kernels
Sometimes kernel is only a single thing on API level. It is convenient for users, but on a particular implementation side it would be better to have multiple kernels (a subgraph) doing this instead. `goodFeaturesToTrack()` can be used as example – while in OpenCV backend it may remain a single kernel, with Fluid it becomes compound. Fluid can handle Harris response calculation but can't do sparse non-maxima suppression and point extraction to an STL vector:

A compound kernel implementation can be defined using a generic macro `GAPI_COMPOUND_KERNEL()`:

```cpp
#include <opencv2/gapi/gcompoundkernel.hpp>       // GAPI_COMPOUND_KERNEL()
using PointArray2f = cv::GArray<cv::Point2f>;
G_TYPED_KERNEL(HarrisCorners,
               <PointArray2f(cv::GMat,int,double,double,int,double)>,
               "org.opencv.imgproc.harris_corner")
{
    static cv::GArrayDesc outMeta(const cv::GMatDesc &,
                                  int,
                                  double,
                                  double,
                                  int,
                                  double)
    {
        // No special metadata for arrays in G-API (yet)
        return cv::empty_array_desc();
    }
};
// Define Fluid-backend-local kernels which form GoodFeatures
G_TYPED_KERNEL(HarrisResponse,
               <cv::GMat(cv::GMat,double,int,double)>,
               "org.opencv.fluid.harris_response")
{
    static cv::GMatDesc outMeta(const cv::GMatDesc &in,
                                double,
                                int,
                                double)
    {
        return in.withType(CV_32F, 1);
    }
};
G_TYPED_KERNEL(ArrayNMS,
               <PointArray2f(cv::GMat,int,double)>,
               "org.opencv.cpu.nms_array")
{
    static cv::GArrayDesc outMeta(const cv::GMatDesc &,
                                  int,
                                  double)
    {
        return cv::empty_array_desc();
    }
};
GAPI_COMPOUND_KERNEL(GFluidHarrisCorners, HarrisCorners)
{
    static PointArray2f
    expand(cv::GMat in,
           int      maxCorners,
           double   quality,
           double   minDist,
           int      blockSize,
           double   k)
    {
        cv::GMat response = HarrisResponse::on(in, quality, blockSize, k);
        return ArrayNMS::on(response, maxCorners, minDist);
    }
};
// Then implement HarrisResponse as Fluid kernel and NMSresponse
// as a generic (OpenCV) kernel
```
It is important to distinguish a compound kernel from G-API high-order function, i.e. a C++ function which looks like a kernel but in fact generates a subgraph. The core difference is that a compound kernel is an *implementation detail* and a kernel implementation may be either compound or not (depending on backend capabilities), while a high-order function is a "macro" in terms of G-API and so cannot act as an interface which then needs to be implemented by a backend.