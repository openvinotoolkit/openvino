/*****************************************************************************\

Copyright 2016 Intel Corporation All Rights Reserved.

The source code contained or described herein and all documents related to
the source code ("Material") are owned by Intel Corporation or its suppliers
or licensors. Title to the Material remains with Intel Corporation or its
suppliers and licensors. The Material contains trade secrets and proprietary
and confidential information of Intel or its suppliers and licensors. The
Material is protected by worldwide copyright and trade secret laws and
treaty provisions. No part of the Material may be used, copied, reproduced,
modified, published, uploaded, posted, transmitted, distributed, or
disclosed in any way without Intel's prior express written permission.

No license under any patent, copyright, trade secret or other intellectual
property right is granted to or conferred upon you by disclosure or delivery
of the Materials, either expressly, by implication, inducement, estoppel or
otherwise. Any license under such intellectual property rights must be
express and approved by Intel in writing.

File Name: cl_intel_planar_yuv.h

Abstract:

Notes:

\*****************************************************************************/
#ifndef __CL_EXT_INTEL_PLANAR_YUV_HPP_
#define __CL_EXT_INTEL_PLANAR_YUV_HPP_

/***************************************
* cl_intel_nv12 extension *
****************************************/
#define CL_NV12_INTEL                           0x410E

#define CL_MEM_NO_ACCESS_INTEL                  ( 1 << 24 )
#define CL_MEM_ACCESS_FLAGS_UNRESTRICTED_INTEL  ( 1 << 25 )

#define CL_DEVICE_PLANAR_YUV_MAX_WIDTH_INTEL    0x417E
#define CL_DEVICE_PLANAR_YUV_MAX_HEIGHT_INTEL   0x417F

#ifdef __cplusplus

#include <CL/cl.hpp>

#ifndef __ERR_STR
#if defined(__CL_ENABLE_EXCEPTIONS)
#define __ERR_STR(x) #x
#else
#define __ERR_STR(x) NULL
#endif // __CL_ENABLE_EXCEPTIONS
#endif

// To avoid accidentally taking ownership of core OpenCL types
// such as cl_kernel constructors are made explicit
// under OpenCL 1.2
#ifndef __CL_EXPLICIT_CONSTRUCTORS
#if defined(CL_VERSION_1_2) && !defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
#define __CL_EXPLICIT_CONSTRUCTORS explicit
#else // #if defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
#define __CL_EXPLICIT_CONSTRUCTORS 
#endif // #if defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
#endif

namespace cl {

class Image2DYPlane : public Image
{
public:
    /*! \brief Constructs a 1D Image in a specified context.
     *
     *  Wraps clCreateImage().
     */
    Image2DYPlane(
        const Context& context,
        cl_mem_flags flags,
        cl_mem parent_nv12_mem_object,
        cl_int* err = NULL)
    {
        cl_int error;
        bool useCreateImage;
		cl::ImageFormat format(CL_R, CL_UNORM_INT8);

#if defined(CL_VERSION_1_2) && defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
        // Run-time decision based on the actual platform
        {
            cl_uint version = detail::getContextPlatformVersion(context());
            useCreateImage = (version >= 0x10002); // OpenCL 1.2 or above
        }
#elif defined(CL_VERSION_1_2)
        useCreateImage = true;
#else
        useCreateImage = false;
#endif

#if defined(CL_VERSION_1_2)
        if (useCreateImage)
        {
            cl_image_desc desc =
            {
                CL_MEM_OBJECT_IMAGE2D,
                0, // width (unused)
                0, // height (unused)
                0, // depth set to y-plane index
				0, // array size (unused)
                0, // row_pitch (unused)
                0, 0, 0, 
				parent_nv12_mem_object
            };
            object_ = ::clCreateImage(
                context(),
                flags,
                &format,
                &desc,
                NULL,
                &error);

            detail::errHandler(error, __CREATE_IMAGE_ERR);
            if (err != NULL) {
                *err = error;
            }
        }
#endif // #if defined(CL_VERSION_1_2)
#if !defined(CL_VERSION_1_2) || defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
        if (!useCreateImage)
        {
            object_ = ::clCreateImage2D(
                context(), flags,&format, width, height, row_pitch, parent_nv12_mem_object, &error);

            detail::errHandler(error, __CREATE_IMAGE2D_ERR);
            if (err != NULL) {
                *err = error;
            }
        }
#endif // #if !defined(CL_VERSION_1_2) || defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
    }

    //! \brief Default constructor - initializes to NULL.
    Image2DYPlane() { }

    /*! \brief Constructor from cl_mem - takes ownership.
     *
     *  See Memory for further details.
     */

    __CL_EXPLICIT_CONSTRUCTORS Image2DYPlane(const cl_mem& image2DYPlane) : Image(image2DYPlane) { }

    /*! \brief Assignment from cl_mem - performs shallow copy.
     *
     *  See Memory for further details.
     */
    Image2DYPlane& operator = (const cl_mem& rhs)
    {
        Image::operator=(rhs);
        return *this;
    }

    /*! \brief Copy constructor to forward copy to the superclass correctly.
     * Required for MSVC.
     */
    Image2DYPlane(const Image2DYPlane& img) : Image(img) {}

    /*! \brief Copy assignment to forward copy to the superclass correctly.
     * Required for MSVC.
     */
    Image2DYPlane& operator = (const Image2DYPlane &img)
    {
        Image::operator=(img);
        return *this;
    }

#if defined(CL_HPP_RVALUE_REFERENCES_SUPPORTED)
    /*! \brief Move constructor to forward move to the superclass correctly.
     * Required for MSVC.
     */
    Image2DYPlane(Image2DYPlane&& img) CL_HPP_NOEXCEPT : Image(std::move(img)) {}

    /*! \brief Move assignment to forward move to the superclass correctly.
     * Required for MSVC.
     */
    Image2DYPlane& operator = (Image2DYPlane &&img)
    {
        Image::operator=(std::move(img));
        return *this;
    }
#endif // #if defined(CL_HPP_RVALUE_REFERENCES_SUPPORTED)
};

class Image2DUVPlane : public Image
{
public:
    /*! \brief Constructs a 1D Image in a specified context.
     *
     *  Wraps clCreateImage().
     */
    Image2DUVPlane(
        const Context& context,
        cl_mem_flags flags,
        cl_mem parent_nv12_mem_object,
        cl_int* err = NULL)
    {
        cl_int error;
        bool useCreateImage;
		cl::ImageFormat format(CL_RG, CL_UNORM_INT8);

#if defined(CL_VERSION_1_2) && defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
        // Run-time decision based on the actual platform
        {
            cl_uint version = detail::getContextPlatformVersion(context());
            useCreateImage = (version >= 0x10002); // OpenCL 1.2 or above
        }
#elif defined(CL_VERSION_1_2)
        useCreateImage = true;
#else
        useCreateImage = false;
#endif

#if defined(CL_VERSION_1_2)
        if (useCreateImage)
        {
            cl_image_desc desc =
            {
                CL_MEM_OBJECT_IMAGE2D,
                0, // width (unused)
                0, // height (unused)
                1, // depth set to uv-plane index
				0, // array size (unused)
                0, // row_pitch (unused)
                0, 0, 0,
				parent_nv12_mem_object
            };
            object_ = ::clCreateImage(
                context(),
                flags,
                &format,
                &desc,
                NULL,
                &error);

            detail::errHandler(error, __CREATE_IMAGE_ERR);
            if (err != NULL) {
                *err = error;
            }
        }
#endif // #if defined(CL_VERSION_1_2)
#if !defined(CL_VERSION_1_2) || defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
        if (!useCreateImage)
        {
            object_ = ::clCreateImage2D(
                context(), flags,&format, width, height, row_pitch, parent_nv12_mem_object, &error);

            detail::errHandler(error, __CREATE_IMAGE2D_ERR);
            if (err != NULL) {
                *err = error;
            }
        }
#endif // #if !defined(CL_VERSION_1_2) || defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
    }

    //! \brief Default constructor - initializes to NULL.
    Image2DUVPlane() { }

    /*! \brief Constructor from cl_mem - takes ownership.
     *
     *  See Memory for further details.
     */
    __CL_EXPLICIT_CONSTRUCTORS Image2DUVPlane(const cl_mem& image2DUVPlane) : Image(image2DUVPlane) { }

    /*! \brief Assignment from cl_mem - performs shallow copy.
     *
     *  See Memory for further details.
     */
    Image2DUVPlane& operator = (const cl_mem& rhs)
    {
        Image::operator=(rhs);
        return *this;
    }

    /*! \brief Copy constructor to forward copy to the superclass correctly.
     * Required for MSVC.
     */
    Image2DUVPlane(const Image2DUVPlane& img) : Image(img) {}

    /*! \brief Copy assignment to forward copy to the superclass correctly.
     * Required for MSVC.
     */
    Image2DUVPlane& operator = (const Image2DUVPlane &img)
    {
        Image::operator=(img);
        return *this;
    }

#if defined(CL_HPP_RVALUE_REFERENCES_SUPPORTED)
    /*! \brief Move constructor to forward move to the superclass correctly.
     * Required for MSVC.
     */
    Image2DUVPlane(Image2DUVPlane&& img) CL_HPP_NOEXCEPT : Image(std::move(img)) {}

    /*! \brief Move assignment to forward move to the superclass correctly.
     * Required for MSVC.
     */
    Image2DUVPlane& operator = (Image2DUVPlane &&img)
    {
        Image::operator=(std::move(img));
        return *this;
    }
#endif // #if defined(CL_HPP_RVALUE_REFERENCES_SUPPORTED)
};

};

#endif


#endif /* __CL_EXT_INTEL_PLANAR_YUV_HPP_ */
