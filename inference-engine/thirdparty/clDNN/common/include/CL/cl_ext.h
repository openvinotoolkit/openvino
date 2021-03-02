/*******************************************************************************
 * Copyright (c) 2008-2020 The Khronos Group Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

/* cl_ext.h contains OpenCL extensions which don't have external */
/* (OpenGL, D3D) dependencies.                                   */

#ifndef __CL_EXT_H
#define __CL_EXT_H

#ifdef __cplusplus
extern "C" {
#endif

#include <CL/cl.h>

/* cl_khr_fp64 extension - no extension #define since it has no functions  */
/* CL_DEVICE_DOUBLE_FP_CONFIG is defined in CL.h for OpenCL >= 120 */

#if CL_TARGET_OPENCL_VERSION <= 110
#define CL_DEVICE_DOUBLE_FP_CONFIG                       0x1032
#endif

/* cl_khr_fp16 extension - no extension #define since it has no functions  */
#define CL_DEVICE_HALF_FP_CONFIG                    0x1033

/* Memory object destruction
 *
 * Apple extension for use to manage externally allocated buffers used with cl_mem objects with CL_MEM_USE_HOST_PTR
 *
 * Registers a user callback function that will be called when the memory object is deleted and its resources
 * freed. Each call to clSetMemObjectCallbackFn registers the specified user callback function on a callback
 * stack associated with memobj. The registered user callback functions are called in the reverse order in
 * which they were registered. The user callback functions are called and then the memory object is deleted
 * and its resources freed. This provides a mechanism for the application (and libraries) using memobj to be
 * notified when the memory referenced by host_ptr, specified when the memory object is created and used as
 * the storage bits for the memory object, can be reused or freed.
 *
 * The application may not call CL api's with the cl_mem object passed to the pfn_notify.
 *
 * Please check for the "cl_APPLE_SetMemObjectDestructor" extension using clGetDeviceInfo(CL_DEVICE_EXTENSIONS)
 * before using.
 */
#define cl_APPLE_SetMemObjectDestructor 1
cl_int  CL_API_ENTRY clSetMemObjectDestructorAPPLE(  cl_mem memobj,
                                        void (* pfn_notify)(cl_mem memobj, void * user_data),
                                        void * user_data)             CL_EXT_SUFFIX__VERSION_1_0;


/* Context Logging Functions
 *
 * The next three convenience functions are intended to be used as the pfn_notify parameter to clCreateContext().
 * Please check for the "cl_APPLE_ContextLoggingFunctions" extension using clGetDeviceInfo(CL_DEVICE_EXTENSIONS)
 * before using.
 *
 * clLogMessagesToSystemLog forwards on all log messages to the Apple System Logger
 */
#define cl_APPLE_ContextLoggingFunctions 1
extern void CL_API_ENTRY clLogMessagesToSystemLogAPPLE(  const char * errstr,
                                            const void * private_info,
                                            size_t       cb,
                                            void *       user_data)  CL_EXT_SUFFIX__VERSION_1_0;

/* clLogMessagesToStdout sends all log messages to the file descriptor stdout */
extern void CL_API_ENTRY clLogMessagesToStdoutAPPLE(   const char * errstr,
                                          const void * private_info,
                                          size_t       cb,
                                          void *       user_data)    CL_EXT_SUFFIX__VERSION_1_0;

/* clLogMessagesToStderr sends all log messages to the file descriptor stderr */
extern void CL_API_ENTRY clLogMessagesToStderrAPPLE(   const char * errstr,
                                          const void * private_info,
                                          size_t       cb,
                                          void *       user_data)    CL_EXT_SUFFIX__VERSION_1_0;


/************************
* cl_khr_icd extension *
************************/
#define cl_khr_icd 1

/* cl_platform_info                                                        */
#define CL_PLATFORM_ICD_SUFFIX_KHR                  0x0920

/* Additional Error Codes                                                  */
#define CL_PLATFORM_NOT_FOUND_KHR                   -1001

extern CL_API_ENTRY cl_int CL_API_CALL
clIcdGetPlatformIDsKHR(cl_uint          num_entries,
                       cl_platform_id * platforms,
                       cl_uint *        num_platforms);

typedef CL_API_ENTRY cl_int
(CL_API_CALL *clIcdGetPlatformIDsKHR_fn)(cl_uint          num_entries,
                                         cl_platform_id * platforms,
                                         cl_uint *        num_platforms);


/*******************************
 * cl_khr_il_program extension *
 *******************************/
#define cl_khr_il_program 1

/* New property to clGetDeviceInfo for retrieving supported intermediate
 * languages
 */
#define CL_DEVICE_IL_VERSION_KHR                    0x105B

/* New property to clGetProgramInfo for retrieving for retrieving the IL of a
 * program
 */
#define CL_PROGRAM_IL_KHR                           0x1169

extern CL_API_ENTRY cl_program CL_API_CALL
clCreateProgramWithILKHR(cl_context   context,
                         const void * il,
                         size_t       length,
                         cl_int *     errcode_ret);

typedef CL_API_ENTRY cl_program
(CL_API_CALL *clCreateProgramWithILKHR_fn)(cl_context   context,
                                           const void * il,
                                           size_t       length,
                                           cl_int *     errcode_ret) CL_EXT_SUFFIX__VERSION_1_2;

/* Extension: cl_khr_image2d_from_buffer
 *
 * This extension allows a 2D image to be created from a cl_mem buffer without
 * a copy. The type associated with a 2D image created from a buffer in an
 * OpenCL program is image2d_t. Both the sampler and sampler-less read_image
 * built-in functions are supported for 2D images and 2D images created from
 * a buffer.  Similarly, the write_image built-ins are also supported for 2D
 * images created from a buffer.
 *
 * When the 2D image from buffer is created, the client must specify the
 * width, height, image format (i.e. channel order and channel data type)
 * and optionally the row pitch.
 *
 * The pitch specified must be a multiple of
 * CL_DEVICE_IMAGE_PITCH_ALIGNMENT_KHR pixels.
 * The base address of the buffer must be aligned to
 * CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT_KHR pixels.
 */

#define CL_DEVICE_IMAGE_PITCH_ALIGNMENT_KHR              0x104A
#define CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT_KHR       0x104B


/**************************************
 * cl_khr_initialize_memory extension *
 **************************************/

#define CL_CONTEXT_MEMORY_INITIALIZE_KHR            0x2030


/**************************************
 * cl_khr_terminate_context extension *
 **************************************/

#define CL_CONTEXT_TERMINATED_KHR                   -1121

#define CL_DEVICE_TERMINATE_CAPABILITY_KHR          0x2031
#define CL_CONTEXT_TERMINATE_KHR                    0x2032

#define cl_khr_terminate_context 1
extern CL_API_ENTRY cl_int CL_API_CALL
clTerminateContextKHR(cl_context context) CL_EXT_SUFFIX__VERSION_1_2;

typedef CL_API_ENTRY cl_int
(CL_API_CALL *clTerminateContextKHR_fn)(cl_context context) CL_EXT_SUFFIX__VERSION_1_2;


/*
 * Extension: cl_khr_spir
 *
 * This extension adds support to create an OpenCL program object from a
 * Standard Portable Intermediate Representation (SPIR) instance
 */

#define CL_DEVICE_SPIR_VERSIONS                     0x40E0
#define CL_PROGRAM_BINARY_TYPE_INTERMEDIATE         0x40E1


/*****************************************
 * cl_khr_create_command_queue extension *
 *****************************************/
#define cl_khr_create_command_queue 1

typedef cl_properties cl_queue_properties_khr;

extern CL_API_ENTRY cl_command_queue CL_API_CALL
clCreateCommandQueueWithPropertiesKHR(cl_context context,
                                      cl_device_id device,
                                      const cl_queue_properties_khr* properties,
                                      cl_int* errcode_ret) CL_EXT_SUFFIX__VERSION_1_2;

typedef CL_API_ENTRY cl_command_queue
(CL_API_CALL *clCreateCommandQueueWithPropertiesKHR_fn)(cl_context context,
                                                        cl_device_id device,
                                                        const cl_queue_properties_khr* properties,
                                                        cl_int* errcode_ret) CL_EXT_SUFFIX__VERSION_1_2;


/******************************************
* cl_nv_device_attribute_query extension *
******************************************/

/* cl_nv_device_attribute_query extension - no extension #define since it has no functions */
#define CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV       0x4000
#define CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV       0x4001
#define CL_DEVICE_REGISTERS_PER_BLOCK_NV            0x4002
#define CL_DEVICE_WARP_SIZE_NV                      0x4003
#define CL_DEVICE_GPU_OVERLAP_NV                    0x4004
#define CL_DEVICE_KERNEL_EXEC_TIMEOUT_NV            0x4005
#define CL_DEVICE_INTEGRATED_MEMORY_NV              0x4006


/*********************************
* cl_amd_device_attribute_query *
*********************************/

#define CL_DEVICE_PROFILING_TIMER_OFFSET_AMD            0x4036
#define CL_DEVICE_TOPOLOGY_AMD                          0x4037
#define CL_DEVICE_BOARD_NAME_AMD                        0x4038
#define CL_DEVICE_GLOBAL_FREE_MEMORY_AMD                0x4039
#define CL_DEVICE_SIMD_PER_COMPUTE_UNIT_AMD             0x4040
#define CL_DEVICE_SIMD_WIDTH_AMD                        0x4041
#define CL_DEVICE_SIMD_INSTRUCTION_WIDTH_AMD            0x4042
#define CL_DEVICE_WAVEFRONT_WIDTH_AMD                   0x4043
#define CL_DEVICE_GLOBAL_MEM_CHANNELS_AMD               0x4044
#define CL_DEVICE_GLOBAL_MEM_CHANNEL_BANKS_AMD          0x4045
#define CL_DEVICE_GLOBAL_MEM_CHANNEL_BANK_WIDTH_AMD     0x4046
#define CL_DEVICE_LOCAL_MEM_SIZE_PER_COMPUTE_UNIT_AMD   0x4047
#define CL_DEVICE_LOCAL_MEM_BANKS_AMD                   0x4048
#define CL_DEVICE_THREAD_TRACE_SUPPORTED_AMD            0x4049
#define CL_DEVICE_GFXIP_MAJOR_AMD                       0x404A
#define CL_DEVICE_GFXIP_MINOR_AMD                       0x404B
#define CL_DEVICE_AVAILABLE_ASYNC_QUEUES_AMD            0x404C
#define CL_DEVICE_PREFERRED_WORK_GROUP_SIZE_AMD         0x4030
#define CL_DEVICE_MAX_WORK_GROUP_SIZE_AMD               0x4031
#define CL_DEVICE_PREFERRED_CONSTANT_BUFFER_SIZE_AMD    0x4033
#define CL_DEVICE_PCIE_ID_AMD                           0x4034


/*********************************
* cl_arm_printf extension
*********************************/

#define CL_PRINTF_CALLBACK_ARM                      0x40B0
#define CL_PRINTF_BUFFERSIZE_ARM                    0x40B1


/***********************************
* cl_ext_device_fission extension
***********************************/
#define cl_ext_device_fission   1

extern CL_API_ENTRY cl_int CL_API_CALL
clReleaseDeviceEXT(cl_device_id device) CL_EXT_SUFFIX__VERSION_1_1;

typedef CL_API_ENTRY cl_int
(CL_API_CALL *clReleaseDeviceEXT_fn)(cl_device_id device) CL_EXT_SUFFIX__VERSION_1_1;

extern CL_API_ENTRY cl_int CL_API_CALL
clRetainDeviceEXT(cl_device_id device) CL_EXT_SUFFIX__VERSION_1_1;

typedef CL_API_ENTRY cl_int
(CL_API_CALL *clRetainDeviceEXT_fn)(cl_device_id device) CL_EXT_SUFFIX__VERSION_1_1;

typedef cl_ulong  cl_device_partition_property_ext;
extern CL_API_ENTRY cl_int CL_API_CALL
clCreateSubDevicesEXT(cl_device_id   in_device,
                      const cl_device_partition_property_ext * properties,
                      cl_uint        num_entries,
                      cl_device_id * out_devices,
                      cl_uint *      num_devices) CL_EXT_SUFFIX__VERSION_1_1;

typedef CL_API_ENTRY cl_int
(CL_API_CALL * clCreateSubDevicesEXT_fn)(cl_device_id   in_device,
                                         const cl_device_partition_property_ext * properties,
                                         cl_uint        num_entries,
                                         cl_device_id * out_devices,
                                         cl_uint *      num_devices) CL_EXT_SUFFIX__VERSION_1_1;

/* cl_device_partition_property_ext */
#define CL_DEVICE_PARTITION_EQUALLY_EXT             0x4050
#define CL_DEVICE_PARTITION_BY_COUNTS_EXT           0x4051
#define CL_DEVICE_PARTITION_BY_NAMES_EXT            0x4052
#define CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN_EXT  0x4053

/* clDeviceGetInfo selectors */
#define CL_DEVICE_PARENT_DEVICE_EXT                 0x4054
#define CL_DEVICE_PARTITION_TYPES_EXT               0x4055
#define CL_DEVICE_AFFINITY_DOMAINS_EXT              0x4056
#define CL_DEVICE_REFERENCE_COUNT_EXT               0x4057
#define CL_DEVICE_PARTITION_STYLE_EXT               0x4058

/* error codes */
#define CL_DEVICE_PARTITION_FAILED_EXT              -1057
#define CL_INVALID_PARTITION_COUNT_EXT              -1058
#define CL_INVALID_PARTITION_NAME_EXT               -1059

/* CL_AFFINITY_DOMAINs */
#define CL_AFFINITY_DOMAIN_L1_CACHE_EXT             0x1
#define CL_AFFINITY_DOMAIN_L2_CACHE_EXT             0x2
#define CL_AFFINITY_DOMAIN_L3_CACHE_EXT             0x3
#define CL_AFFINITY_DOMAIN_L4_CACHE_EXT             0x4
#define CL_AFFINITY_DOMAIN_NUMA_EXT                 0x10
#define CL_AFFINITY_DOMAIN_NEXT_FISSIONABLE_EXT     0x100

/* cl_device_partition_property_ext list terminators */
#define CL_PROPERTIES_LIST_END_EXT                  ((cl_device_partition_property_ext) 0)
#define CL_PARTITION_BY_COUNTS_LIST_END_EXT         ((cl_device_partition_property_ext) 0)
#define CL_PARTITION_BY_NAMES_LIST_END_EXT          ((cl_device_partition_property_ext) 0 - 1)


/***********************************
 * cl_ext_migrate_memobject extension definitions
 ***********************************/
#define cl_ext_migrate_memobject 1

typedef cl_bitfield cl_mem_migration_flags_ext;

#define CL_MIGRATE_MEM_OBJECT_HOST_EXT              0x1

#define CL_COMMAND_MIGRATE_MEM_OBJECT_EXT           0x4040

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueMigrateMemObjectEXT(cl_command_queue command_queue,
                             cl_uint          num_mem_objects,
                             const cl_mem *   mem_objects,
                             cl_mem_migration_flags_ext flags,
                             cl_uint          num_events_in_wait_list,
                             const cl_event * event_wait_list,
                             cl_event *       event);

typedef CL_API_ENTRY cl_int
(CL_API_CALL *clEnqueueMigrateMemObjectEXT_fn)(cl_command_queue command_queue,
                                               cl_uint          num_mem_objects,
                                               const cl_mem *   mem_objects,
                                               cl_mem_migration_flags_ext flags,
                                               cl_uint          num_events_in_wait_list,
                                               const cl_event * event_wait_list,
                                               cl_event *       event);


/*********************************
* cl_ext_cxx_for_opencl extension
*********************************/
#define cl_ext_cxx_for_opencl 1

#define CL_DEVICE_CXX_FOR_OPENCL_NUMERIC_VERSION_EXT 0x4230

/*********************************
* cl_qcom_ext_host_ptr extension
*********************************/
#define cl_qcom_ext_host_ptr 1

#define CL_MEM_EXT_HOST_PTR_QCOM                  (1 << 29)

#define CL_DEVICE_EXT_MEM_PADDING_IN_BYTES_QCOM   0x40A0
#define CL_DEVICE_PAGE_SIZE_QCOM                  0x40A1
#define CL_IMAGE_ROW_ALIGNMENT_QCOM               0x40A2
#define CL_IMAGE_SLICE_ALIGNMENT_QCOM             0x40A3
#define CL_MEM_HOST_UNCACHED_QCOM                 0x40A4
#define CL_MEM_HOST_WRITEBACK_QCOM                0x40A5
#define CL_MEM_HOST_WRITETHROUGH_QCOM             0x40A6
#define CL_MEM_HOST_WRITE_COMBINING_QCOM          0x40A7

typedef cl_uint                                   cl_image_pitch_info_qcom;

extern CL_API_ENTRY cl_int CL_API_CALL
clGetDeviceImageInfoQCOM(cl_device_id             device,
                         size_t                   image_width,
                         size_t                   image_height,
                         const cl_image_format   *image_format,
                         cl_image_pitch_info_qcom param_name,
                         size_t                   param_value_size,
                         void                    *param_value,
                         size_t                  *param_value_size_ret);

typedef struct _cl_mem_ext_host_ptr
{
    /* Type of external memory allocation. */
    /* Legal values will be defined in layered extensions. */
    cl_uint  allocation_type;

    /* Host cache policy for this external memory allocation. */
    cl_uint  host_cache_policy;

} cl_mem_ext_host_ptr;


/*******************************************
* cl_qcom_ext_host_ptr_iocoherent extension
********************************************/

/* Cache policy specifying io-coherence */
#define CL_MEM_HOST_IOCOHERENT_QCOM               0x40A9


/*********************************
* cl_qcom_ion_host_ptr extension
*********************************/

#define CL_MEM_ION_HOST_PTR_QCOM                  0x40A8

typedef struct _cl_mem_ion_host_ptr
{
    /* Type of external memory allocation. */
    /* Must be CL_MEM_ION_HOST_PTR_QCOM for ION allocations. */
    cl_mem_ext_host_ptr  ext_host_ptr;

    /* ION file descriptor */
    int                  ion_filedesc;

    /* Host pointer to the ION allocated memory */
    void*                ion_hostptr;

} cl_mem_ion_host_ptr;


/*********************************
* cl_qcom_android_native_buffer_host_ptr extension
*********************************/

#define CL_MEM_ANDROID_NATIVE_BUFFER_HOST_PTR_QCOM                  0x40C6

typedef struct _cl_mem_android_native_buffer_host_ptr
{
    /* Type of external memory allocation. */
    /* Must be CL_MEM_ANDROID_NATIVE_BUFFER_HOST_PTR_QCOM for Android native buffers. */
    cl_mem_ext_host_ptr  ext_host_ptr;

    /* Virtual pointer to the android native buffer */
    void*                anb_ptr;

} cl_mem_android_native_buffer_host_ptr;


/******************************************
 * cl_img_yuv_image extension *
 ******************************************/

/* Image formats used in clCreateImage */
#define CL_NV21_IMG                                 0x40D0
#define CL_YV12_IMG                                 0x40D1


/******************************************
 * cl_img_cached_allocations extension *
 ******************************************/

/* Flag values used by clCreateBuffer */
#define CL_MEM_USE_UNCACHED_CPU_MEMORY_IMG          (1 << 26)
#define CL_MEM_USE_CACHED_CPU_MEMORY_IMG            (1 << 27)


/******************************************
 * cl_img_use_gralloc_ptr extension *
 ******************************************/
#define cl_img_use_gralloc_ptr 1

/* Flag values used by clCreateBuffer */
#define CL_MEM_USE_GRALLOC_PTR_IMG                  (1 << 28)

/* To be used by clGetEventInfo: */
#define CL_COMMAND_ACQUIRE_GRALLOC_OBJECTS_IMG      0x40D2
#define CL_COMMAND_RELEASE_GRALLOC_OBJECTS_IMG      0x40D3

/* Error codes from clEnqueueAcquireGrallocObjectsIMG and clEnqueueReleaseGrallocObjectsIMG */
#define CL_GRALLOC_RESOURCE_NOT_ACQUIRED_IMG        0x40D4
#define CL_INVALID_GRALLOC_OBJECT_IMG               0x40D5

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueAcquireGrallocObjectsIMG(cl_command_queue      command_queue,
                                  cl_uint               num_objects,
                                  const cl_mem *        mem_objects,
                                  cl_uint               num_events_in_wait_list,
                                  const cl_event *      event_wait_list,
                                  cl_event *            event) CL_EXT_SUFFIX__VERSION_1_2;

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueReleaseGrallocObjectsIMG(cl_command_queue      command_queue,
                                  cl_uint               num_objects,
                                  const cl_mem *        mem_objects,
                                  cl_uint               num_events_in_wait_list,
                                  const cl_event *      event_wait_list,
                                  cl_event *            event) CL_EXT_SUFFIX__VERSION_1_2;

/******************************************
 * cl_img_generate_mipmap extension *
 ******************************************/
#define cl_img_generate_mipmap 1

typedef cl_uint cl_mipmap_filter_mode_img;

/* To be used by clEnqueueGenerateMipmapIMG */
#define CL_MIPMAP_FILTER_ANY_IMG 0x0
#define CL_MIPMAP_FILTER_BOX_IMG 0x1

/* To be used by clGetEventInfo */
#define CL_COMMAND_GENERATE_MIPMAP_IMG 0x40D6

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueGenerateMipmapIMG(cl_command_queue          command_queue,
                           cl_mem                    src_image,
                           cl_mem                    dst_image,
                           cl_mipmap_filter_mode_img mipmap_filter_mode,
                           const size_t              *array_region,
                           const size_t              *mip_region,
                           cl_uint                   num_events_in_wait_list,
                           const cl_event            *event_wait_list,
                           cl_event *event) CL_EXT_SUFFIX__VERSION_1_2;
  
/******************************************
 * cl_img_mem_properties extension *
 ******************************************/
#define cl_img_mem_properties 1

/* To be used by clCreateBufferWithProperties */
#define CL_MEM_ALLOC_FLAGS_IMG 0x40D7

/* To be used wiith the CL_MEM_ALLOC_FLAGS_IMG property */
typedef cl_bitfield cl_mem_alloc_flags_img;

/* To be used with cl_mem_alloc_flags_img */
#define CL_MEM_ALLOC_RELAX_REQUIREMENTS_IMG (1 << 0)

/*********************************
* cl_khr_subgroups extension
*********************************/
#define cl_khr_subgroups 1

#if !defined(CL_VERSION_2_1)
/* For OpenCL 2.1 and newer, cl_kernel_sub_group_info is declared in CL.h.
   In hindsight, there should have been a khr suffix on this type for
   the extension, but keeping it un-suffixed to maintain backwards
   compatibility. */
typedef cl_uint             cl_kernel_sub_group_info;
#endif

/* cl_kernel_sub_group_info */
#define CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE_KHR    0x2033
#define CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE_KHR       0x2034

extern CL_API_ENTRY cl_int CL_API_CALL
clGetKernelSubGroupInfoKHR(cl_kernel    in_kernel,
                           cl_device_id in_device,
                           cl_kernel_sub_group_info param_name,
                           size_t       input_value_size,
                           const void * input_value,
                           size_t       param_value_size,
                           void *       param_value,
                           size_t *     param_value_size_ret) CL_EXT_SUFFIX__VERSION_2_0_DEPRECATED;

typedef CL_API_ENTRY cl_int
(CL_API_CALL * clGetKernelSubGroupInfoKHR_fn)(cl_kernel    in_kernel,
                                              cl_device_id in_device,
                                              cl_kernel_sub_group_info param_name,
                                              size_t       input_value_size,
                                              const void * input_value,
                                              size_t       param_value_size,
                                              void *       param_value,
                                              size_t *     param_value_size_ret) CL_EXT_SUFFIX__VERSION_2_0_DEPRECATED;


/*********************************
* cl_khr_mipmap_image extension
*********************************/

/* cl_sampler_properties */
#define CL_SAMPLER_MIP_FILTER_MODE_KHR              0x1155
#define CL_SAMPLER_LOD_MIN_KHR                      0x1156
#define CL_SAMPLER_LOD_MAX_KHR                      0x1157


/*********************************
* cl_khr_priority_hints extension
*********************************/
/* This extension define is for backwards compatibility.
   It shouldn't be required since this extension has no new functions. */
#define cl_khr_priority_hints 1

typedef cl_uint  cl_queue_priority_khr;

/* cl_command_queue_properties */
#define CL_QUEUE_PRIORITY_KHR 0x1096

/* cl_queue_priority_khr */
#define CL_QUEUE_PRIORITY_HIGH_KHR (1<<0)
#define CL_QUEUE_PRIORITY_MED_KHR (1<<1)
#define CL_QUEUE_PRIORITY_LOW_KHR (1<<2)


/*********************************
* cl_khr_throttle_hints extension
*********************************/
/* This extension define is for backwards compatibility.
   It shouldn't be required since this extension has no new functions. */
#define cl_khr_throttle_hints 1

typedef cl_uint  cl_queue_throttle_khr;

/* cl_command_queue_properties */
#define CL_QUEUE_THROTTLE_KHR 0x1097

/* cl_queue_throttle_khr */
#define CL_QUEUE_THROTTLE_HIGH_KHR (1<<0)
#define CL_QUEUE_THROTTLE_MED_KHR (1<<1)
#define CL_QUEUE_THROTTLE_LOW_KHR (1<<2)


/*********************************
* cl_khr_subgroup_named_barrier
*********************************/
/* This extension define is for backwards compatibility.
   It shouldn't be required since this extension has no new functions. */
#define cl_khr_subgroup_named_barrier 1

/* cl_device_info */
#define CL_DEVICE_MAX_NAMED_BARRIER_COUNT_KHR       0x2035


/*********************************
* cl_khr_extended_versioning
*********************************/

#define cl_khr_extended_versioning 1

#define CL_VERSION_MAJOR_BITS_KHR (10)
#define CL_VERSION_MINOR_BITS_KHR (10)
#define CL_VERSION_PATCH_BITS_KHR (12)

#define CL_VERSION_MAJOR_MASK_KHR ((1 << CL_VERSION_MAJOR_BITS_KHR) - 1)
#define CL_VERSION_MINOR_MASK_KHR ((1 << CL_VERSION_MINOR_BITS_KHR) - 1)
#define CL_VERSION_PATCH_MASK_KHR ((1 << CL_VERSION_PATCH_BITS_KHR) - 1)

#define CL_VERSION_MAJOR_KHR(version) ((version) >> (CL_VERSION_MINOR_BITS_KHR + CL_VERSION_PATCH_BITS_KHR))
#define CL_VERSION_MINOR_KHR(version) (((version) >> CL_VERSION_PATCH_BITS_KHR) & CL_VERSION_MINOR_MASK_KHR)
#define CL_VERSION_PATCH_KHR(version) ((version) & CL_VERSION_PATCH_MASK_KHR)

#define CL_MAKE_VERSION_KHR(major, minor, patch) \
    ((((major) & CL_VERSION_MAJOR_MASK_KHR) << (CL_VERSION_MINOR_BITS_KHR + CL_VERSION_PATCH_BITS_KHR)) | \
    (((minor) &  CL_VERSION_MINOR_MASK_KHR) << CL_VERSION_PATCH_BITS_KHR) | \
    ((patch) & CL_VERSION_PATCH_MASK_KHR))

typedef cl_uint cl_version_khr;

#define CL_NAME_VERSION_MAX_NAME_SIZE_KHR 64

typedef struct _cl_name_version_khr
{
    cl_version_khr version;
    char name[CL_NAME_VERSION_MAX_NAME_SIZE_KHR];
} cl_name_version_khr;

/* cl_platform_info */
#define CL_PLATFORM_NUMERIC_VERSION_KHR                  0x0906
#define CL_PLATFORM_EXTENSIONS_WITH_VERSION_KHR          0x0907

/* cl_device_info */
#define CL_DEVICE_NUMERIC_VERSION_KHR                    0x105E
#define CL_DEVICE_OPENCL_C_NUMERIC_VERSION_KHR           0x105F
#define CL_DEVICE_EXTENSIONS_WITH_VERSION_KHR            0x1060
#define CL_DEVICE_ILS_WITH_VERSION_KHR                   0x1061
#define CL_DEVICE_BUILT_IN_KERNELS_WITH_VERSION_KHR      0x1062


/*********************************
* cl_khr_device_uuid extension
*********************************/
#define cl_khr_device_uuid 1

#define CL_UUID_SIZE_KHR 16
#define CL_LUID_SIZE_KHR 8

#define CL_DEVICE_UUID_KHR          0x106A
#define CL_DRIVER_UUID_KHR          0x106B
#define CL_DEVICE_LUID_VALID_KHR    0x106C
#define CL_DEVICE_LUID_KHR          0x106D
#define CL_DEVICE_NODE_MASK_KHR     0x106E


/**********************************
 * cl_arm_import_memory extension *
 **********************************/
#define cl_arm_import_memory 1

typedef intptr_t cl_import_properties_arm;

/* Default and valid proporties name for cl_arm_import_memory */
#define CL_IMPORT_TYPE_ARM                        0x40B2

/* Host process memory type default value for CL_IMPORT_TYPE_ARM property */
#define CL_IMPORT_TYPE_HOST_ARM                   0x40B3

/* DMA BUF memory type value for CL_IMPORT_TYPE_ARM property */
#define CL_IMPORT_TYPE_DMA_BUF_ARM                0x40B4

/* Protected memory property */
#define CL_IMPORT_TYPE_PROTECTED_ARM              0x40B5

/* Android hardware buffer type value for CL_IMPORT_TYPE_ARM property */
#define CL_IMPORT_TYPE_ANDROID_HARDWARE_BUFFER_ARM 0x41E2

/* Data consistency with host property */
#define CL_IMPORT_DMA_BUF_DATA_CONSISTENCY_WITH_HOST_ARM 0x41E3

/* Import memory size value to indicate a size for the whole buffer */
#define CL_IMPORT_MEMORY_WHOLE_ALLOCATION_ARM SIZE_MAX

/* This extension adds a new function that allows for direct memory import into
 * OpenCL via the clImportMemoryARM function.
 *
 * Memory imported through this interface will be mapped into the device's page
 * tables directly, providing zero copy access. It will never fall back to copy
 * operations and aliased buffers.
 *
 * Types of memory supported for import are specified as additional extension
 * strings.
 *
 * This extension produces cl_mem allocations which are compatible with all other
 * users of cl_mem in the standard API.
 *
 * This extension maps pages with the same properties as the normal buffer creation
 * function clCreateBuffer.
 */
extern CL_API_ENTRY cl_mem CL_API_CALL
clImportMemoryARM( cl_context context,
                   cl_mem_flags flags,
                   const cl_import_properties_arm *properties,
                   void *memory,
                   size_t size,
                   cl_int *errcode_ret) CL_EXT_SUFFIX__VERSION_1_0;


/******************************************
 * cl_arm_shared_virtual_memory extension *
 ******************************************/
#define cl_arm_shared_virtual_memory 1

/* Used by clGetDeviceInfo */
#define CL_DEVICE_SVM_CAPABILITIES_ARM                  0x40B6

/* Used by clGetMemObjectInfo */
#define CL_MEM_USES_SVM_POINTER_ARM                     0x40B7

/* Used by clSetKernelExecInfoARM: */
#define CL_KERNEL_EXEC_INFO_SVM_PTRS_ARM                0x40B8
#define CL_KERNEL_EXEC_INFO_SVM_FINE_GRAIN_SYSTEM_ARM   0x40B9

/* To be used by clGetEventInfo: */
#define CL_COMMAND_SVM_FREE_ARM                         0x40BA
#define CL_COMMAND_SVM_MEMCPY_ARM                       0x40BB
#define CL_COMMAND_SVM_MEMFILL_ARM                      0x40BC
#define CL_COMMAND_SVM_MAP_ARM                          0x40BD
#define CL_COMMAND_SVM_UNMAP_ARM                        0x40BE

/* Flag values returned by clGetDeviceInfo with CL_DEVICE_SVM_CAPABILITIES_ARM as the param_name. */
#define CL_DEVICE_SVM_COARSE_GRAIN_BUFFER_ARM           (1 << 0)
#define CL_DEVICE_SVM_FINE_GRAIN_BUFFER_ARM             (1 << 1)
#define CL_DEVICE_SVM_FINE_GRAIN_SYSTEM_ARM             (1 << 2)
#define CL_DEVICE_SVM_ATOMICS_ARM                       (1 << 3)

/* Flag values used by clSVMAllocARM: */
#define CL_MEM_SVM_FINE_GRAIN_BUFFER_ARM                (1 << 10)
#define CL_MEM_SVM_ATOMICS_ARM                          (1 << 11)

typedef cl_bitfield cl_svm_mem_flags_arm;
typedef cl_uint     cl_kernel_exec_info_arm;
typedef cl_bitfield cl_device_svm_capabilities_arm;

extern CL_API_ENTRY void * CL_API_CALL
clSVMAllocARM(cl_context       context,
              cl_svm_mem_flags_arm flags,
              size_t           size,
              cl_uint          alignment) CL_EXT_SUFFIX__VERSION_1_2;

extern CL_API_ENTRY void CL_API_CALL
clSVMFreeARM(cl_context        context,
             void *            svm_pointer) CL_EXT_SUFFIX__VERSION_1_2;

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueSVMFreeARM(cl_command_queue  command_queue,
                    cl_uint           num_svm_pointers,
                    void *            svm_pointers[],
                    void (CL_CALLBACK * pfn_free_func)(cl_command_queue queue,
                                                       cl_uint          num_svm_pointers,
                                                       void *           svm_pointers[],
                                                       void *           user_data),
                    void *            user_data,
                    cl_uint           num_events_in_wait_list,
                    const cl_event *  event_wait_list,
                    cl_event *        event) CL_EXT_SUFFIX__VERSION_1_2;

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueSVMMemcpyARM(cl_command_queue  command_queue,
                      cl_bool           blocking_copy,
                      void *            dst_ptr,
                      const void *      src_ptr,
                      size_t            size,
                      cl_uint           num_events_in_wait_list,
                      const cl_event *  event_wait_list,
                      cl_event *        event) CL_EXT_SUFFIX__VERSION_1_2;

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueSVMMemFillARM(cl_command_queue  command_queue,
                       void *            svm_ptr,
                       const void *      pattern,
                       size_t            pattern_size,
                       size_t            size,
                       cl_uint           num_events_in_wait_list,
                       const cl_event *  event_wait_list,
                       cl_event *        event) CL_EXT_SUFFIX__VERSION_1_2;

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueSVMMapARM(cl_command_queue  command_queue,
                   cl_bool           blocking_map,
                   cl_map_flags      flags,
                   void *            svm_ptr,
                   size_t            size,
                   cl_uint           num_events_in_wait_list,
                   const cl_event *  event_wait_list,
                   cl_event *        event) CL_EXT_SUFFIX__VERSION_1_2;

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueSVMUnmapARM(cl_command_queue  command_queue,
                     void *            svm_ptr,
                     cl_uint           num_events_in_wait_list,
                     const cl_event *  event_wait_list,
                     cl_event *        event) CL_EXT_SUFFIX__VERSION_1_2;

extern CL_API_ENTRY cl_int CL_API_CALL
clSetKernelArgSVMPointerARM(cl_kernel    kernel,
                            cl_uint      arg_index,
                            const void * arg_value) CL_EXT_SUFFIX__VERSION_1_2;

extern CL_API_ENTRY cl_int CL_API_CALL
clSetKernelExecInfoARM(cl_kernel            kernel,
                       cl_kernel_exec_info_arm  param_name,
                       size_t               param_value_size,
                       const void *         param_value) CL_EXT_SUFFIX__VERSION_1_2;

/********************************
 * cl_arm_get_core_id extension *
 ********************************/

#ifdef CL_VERSION_1_2

#define cl_arm_get_core_id 1

/* Device info property for bitfield of cores present */
#define CL_DEVICE_COMPUTE_UNITS_BITFIELD_ARM      0x40BF

#endif  /* CL_VERSION_1_2 */

/*********************************
* cl_arm_job_slot_selection
*********************************/

#define cl_arm_job_slot_selection 1

/* cl_device_info */
#define CL_DEVICE_JOB_SLOTS_ARM                   0x41E0

/* cl_command_queue_properties */
#define CL_QUEUE_JOB_SLOT_ARM                     0x41E1

/*********************************
* cl_arm_scheduling_controls
*********************************/

#define cl_arm_scheduling_controls 1

typedef cl_bitfield cl_device_scheduling_controls_capabilities_arm;

/* cl_device_info */
#define CL_DEVICE_SCHEDULING_CONTROLS_CAPABILITIES_ARM          0x41E4

#define CL_DEVICE_SCHEDULING_KERNEL_BATCHING_ARM               (1 << 0)
#define CL_DEVICE_SCHEDULING_WORKGROUP_BATCH_SIZE_ARM          (1 << 1)
#define CL_DEVICE_SCHEDULING_WORKGROUP_BATCH_SIZE_MODIFIER_ARM (1 << 2)
#define CL_DEVICE_SCHEDULING_DEFERRED_FLUSH_ARM                (1 << 3)
#define CL_DEVICE_SCHEDULING_REGISTER_ALLOCATION_ARM           (1 << 4)

#define CL_DEVICE_SUPPORTED_REGISTER_ALLOCATIONS_ARM            0x41EB

/* cl_kernel_info */
#define CL_KERNEL_EXEC_INFO_WORKGROUP_BATCH_SIZE_ARM            0x41E5
#define CL_KERNEL_EXEC_INFO_WORKGROUP_BATCH_SIZE_MODIFIER_ARM   0x41E6

/* cl_queue_properties */
#define CL_QUEUE_KERNEL_BATCHING_ARM                            0x41E7
#define CL_QUEUE_DEFERRED_FLUSH_ARM                             0x41EC

/**************************************
* cl_arm_controlled_kernel_termination
***************************************/

#define cl_arm_controlled_kernel_termination 1

/* Error code to indicate kernel terminated with failure */
#define CL_COMMAND_TERMINATED_ITSELF_WITH_FAILURE_ARM -1108

/* cl_device_info */
#define CL_DEVICE_CONTROLLED_TERMINATION_CAPABILITIES_ARM 0x41EE

/* Bit fields for controlled termination feature query */
typedef cl_bitfield cl_device_controlled_termination_capabilities_arm;

#define CL_DEVICE_CONTROLLED_TERMINATION_SUCCESS_ARM (1 << 0)
#define CL_DEVICE_CONTROLLED_TERMINATION_FAILURE_ARM (1 << 1)
#define CL_DEVICE_CONTROLLED_TERMINATION_QUERY_ARM (1 << 2)

/* cl_event_info */
#define CL_EVENT_COMMAND_TERMINATION_REASON_ARM 0x41ED

/* Values returned for event termination reason query */
typedef cl_uint cl_command_termination_reason_arm;

#define CL_COMMAND_TERMINATION_COMPLETION_ARM  0
#define CL_COMMAND_TERMINATION_CONTROLLED_SUCCESS_ARM 1
#define CL_COMMAND_TERMINATION_CONTROLLED_FAILURE_ARM 2
#define CL_COMMAND_TERMINATION_ERROR_ARM 3

/***************************************
* cl_intel_thread_local_exec extension *
****************************************/

#define cl_intel_thread_local_exec 1

#define CL_QUEUE_THREAD_LOCAL_EXEC_ENABLE_INTEL      (((cl_bitfield)1) << 31)

/***********************************************
* cl_intel_device_partition_by_names extension *
************************************************/

#define cl_intel_device_partition_by_names 1

#define CL_DEVICE_PARTITION_BY_NAMES_INTEL          0x4052
#define CL_PARTITION_BY_NAMES_LIST_END_INTEL        -1

/************************************************
* cl_intel_accelerator extension                *
* cl_intel_motion_estimation extension          *
* cl_intel_advanced_motion_estimation extension *
*************************************************/

#define cl_intel_accelerator 1
#define cl_intel_motion_estimation 1
#define cl_intel_advanced_motion_estimation 1

typedef struct _cl_accelerator_intel* cl_accelerator_intel;
typedef cl_uint cl_accelerator_type_intel;
typedef cl_uint cl_accelerator_info_intel;

typedef struct _cl_motion_estimation_desc_intel {
    cl_uint mb_block_type;
    cl_uint subpixel_mode;
    cl_uint sad_adjust_mode;
    cl_uint search_path_type;
} cl_motion_estimation_desc_intel;

/* error codes */
#define CL_INVALID_ACCELERATOR_INTEL                              -1094
#define CL_INVALID_ACCELERATOR_TYPE_INTEL                         -1095
#define CL_INVALID_ACCELERATOR_DESCRIPTOR_INTEL                   -1096
#define CL_ACCELERATOR_TYPE_NOT_SUPPORTED_INTEL                   -1097

/* cl_accelerator_type_intel */
#define CL_ACCELERATOR_TYPE_MOTION_ESTIMATION_INTEL               0x0

/* cl_accelerator_info_intel */
#define CL_ACCELERATOR_DESCRIPTOR_INTEL                           0x4090
#define CL_ACCELERATOR_REFERENCE_COUNT_INTEL                      0x4091
#define CL_ACCELERATOR_CONTEXT_INTEL                              0x4092
#define CL_ACCELERATOR_TYPE_INTEL                                 0x4093

/* cl_motion_detect_desc_intel flags */
#define CL_ME_MB_TYPE_16x16_INTEL                                 0x0
#define CL_ME_MB_TYPE_8x8_INTEL                                   0x1
#define CL_ME_MB_TYPE_4x4_INTEL                                   0x2

#define CL_ME_SUBPIXEL_MODE_INTEGER_INTEL                         0x0
#define CL_ME_SUBPIXEL_MODE_HPEL_INTEL                            0x1
#define CL_ME_SUBPIXEL_MODE_QPEL_INTEL                            0x2

#define CL_ME_SAD_ADJUST_MODE_NONE_INTEL                          0x0
#define CL_ME_SAD_ADJUST_MODE_HAAR_INTEL                          0x1

#define CL_ME_SEARCH_PATH_RADIUS_2_2_INTEL                        0x0
#define CL_ME_SEARCH_PATH_RADIUS_4_4_INTEL                        0x1
#define CL_ME_SEARCH_PATH_RADIUS_16_12_INTEL                      0x5

#define CL_ME_SKIP_BLOCK_TYPE_16x16_INTEL                         0x0
#define CL_ME_CHROMA_INTRA_PREDICT_ENABLED_INTEL                  0x1
#define CL_ME_LUMA_INTRA_PREDICT_ENABLED_INTEL                    0x2
#define CL_ME_SKIP_BLOCK_TYPE_8x8_INTEL                           0x4

#define CL_ME_FORWARD_INPUT_MODE_INTEL                            0x1
#define CL_ME_BACKWARD_INPUT_MODE_INTEL                           0x2
#define CL_ME_BIDIRECTION_INPUT_MODE_INTEL                        0x3

#define CL_ME_BIDIR_WEIGHT_QUARTER_INTEL                          16
#define CL_ME_BIDIR_WEIGHT_THIRD_INTEL                            21
#define CL_ME_BIDIR_WEIGHT_HALF_INTEL                             32
#define CL_ME_BIDIR_WEIGHT_TWO_THIRD_INTEL                        43
#define CL_ME_BIDIR_WEIGHT_THREE_QUARTER_INTEL                    48

#define CL_ME_COST_PENALTY_NONE_INTEL                             0x0
#define CL_ME_COST_PENALTY_LOW_INTEL                              0x1
#define CL_ME_COST_PENALTY_NORMAL_INTEL                           0x2
#define CL_ME_COST_PENALTY_HIGH_INTEL                             0x3

#define CL_ME_COST_PRECISION_QPEL_INTEL                           0x0
#define CL_ME_COST_PRECISION_HPEL_INTEL                           0x1
#define CL_ME_COST_PRECISION_PEL_INTEL                            0x2
#define CL_ME_COST_PRECISION_DPEL_INTEL                           0x3

#define CL_ME_LUMA_PREDICTOR_MODE_VERTICAL_INTEL                  0x0
#define CL_ME_LUMA_PREDICTOR_MODE_HORIZONTAL_INTEL                0x1
#define CL_ME_LUMA_PREDICTOR_MODE_DC_INTEL                        0x2
#define CL_ME_LUMA_PREDICTOR_MODE_DIAGONAL_DOWN_LEFT_INTEL        0x3

#define CL_ME_LUMA_PREDICTOR_MODE_DIAGONAL_DOWN_RIGHT_INTEL       0x4
#define CL_ME_LUMA_PREDICTOR_MODE_PLANE_INTEL                     0x4
#define CL_ME_LUMA_PREDICTOR_MODE_VERTICAL_RIGHT_INTEL            0x5
#define CL_ME_LUMA_PREDICTOR_MODE_HORIZONTAL_DOWN_INTEL           0x6
#define CL_ME_LUMA_PREDICTOR_MODE_VERTICAL_LEFT_INTEL             0x7
#define CL_ME_LUMA_PREDICTOR_MODE_HORIZONTAL_UP_INTEL             0x8

#define CL_ME_CHROMA_PREDICTOR_MODE_DC_INTEL                      0x0
#define CL_ME_CHROMA_PREDICTOR_MODE_HORIZONTAL_INTEL              0x1
#define CL_ME_CHROMA_PREDICTOR_MODE_VERTICAL_INTEL                0x2
#define CL_ME_CHROMA_PREDICTOR_MODE_PLANE_INTEL                   0x3

/* cl_device_info */
#define CL_DEVICE_ME_VERSION_INTEL                                0x407E

#define CL_ME_VERSION_LEGACY_INTEL                                0x0
#define CL_ME_VERSION_ADVANCED_VER_1_INTEL                        0x1
#define CL_ME_VERSION_ADVANCED_VER_2_INTEL                        0x2

extern CL_API_ENTRY cl_accelerator_intel CL_API_CALL
clCreateAcceleratorINTEL(
    cl_context                   context,
    cl_accelerator_type_intel    accelerator_type,
    size_t                       descriptor_size,
    const void*                  descriptor,
    cl_int*                      errcode_ret) CL_EXT_SUFFIX__VERSION_1_2;

typedef CL_API_ENTRY cl_accelerator_intel (CL_API_CALL *clCreateAcceleratorINTEL_fn)(
    cl_context                   context,
    cl_accelerator_type_intel    accelerator_type,
    size_t                       descriptor_size,
    const void*                  descriptor,
    cl_int*                      errcode_ret) CL_EXT_SUFFIX__VERSION_1_2;

extern CL_API_ENTRY cl_int CL_API_CALL
clGetAcceleratorInfoINTEL(
    cl_accelerator_intel         accelerator,
    cl_accelerator_info_intel    param_name,
    size_t                       param_value_size,
    void*                        param_value,
    size_t*                      param_value_size_ret) CL_EXT_SUFFIX__VERSION_1_2;

typedef CL_API_ENTRY cl_int (CL_API_CALL *clGetAcceleratorInfoINTEL_fn)(
    cl_accelerator_intel         accelerator,
    cl_accelerator_info_intel    param_name,
    size_t                       param_value_size,
    void*                        param_value,
    size_t*                      param_value_size_ret) CL_EXT_SUFFIX__VERSION_1_2;

extern CL_API_ENTRY cl_int CL_API_CALL
clRetainAcceleratorINTEL(
    cl_accelerator_intel         accelerator) CL_EXT_SUFFIX__VERSION_1_2;

typedef CL_API_ENTRY cl_int (CL_API_CALL *clRetainAcceleratorINTEL_fn)(
    cl_accelerator_intel         accelerator) CL_EXT_SUFFIX__VERSION_1_2;

extern CL_API_ENTRY cl_int CL_API_CALL
clReleaseAcceleratorINTEL(
    cl_accelerator_intel         accelerator) CL_EXT_SUFFIX__VERSION_1_2;

typedef CL_API_ENTRY cl_int (CL_API_CALL *clReleaseAcceleratorINTEL_fn)(
    cl_accelerator_intel         accelerator) CL_EXT_SUFFIX__VERSION_1_2;

/******************************************
* cl_intel_simultaneous_sharing extension *
*******************************************/

#define cl_intel_simultaneous_sharing 1

#define CL_DEVICE_SIMULTANEOUS_INTEROPS_INTEL            0x4104
#define CL_DEVICE_NUM_SIMULTANEOUS_INTEROPS_INTEL        0x4105

/***********************************
* cl_intel_egl_image_yuv extension *
************************************/

#define cl_intel_egl_image_yuv 1

#define CL_EGL_YUV_PLANE_INTEL                           0x4107

/********************************
* cl_intel_packed_yuv extension *
*********************************/

#define cl_intel_packed_yuv 1

#define CL_YUYV_INTEL                                    0x4076
#define CL_UYVY_INTEL                                    0x4077
#define CL_YVYU_INTEL                                    0x4078
#define CL_VYUY_INTEL                                    0x4079

/********************************************
* cl_intel_required_subgroup_size extension *
*********************************************/

#define cl_intel_required_subgroup_size 1

#define CL_DEVICE_SUB_GROUP_SIZES_INTEL                  0x4108
#define CL_KERNEL_SPILL_MEM_SIZE_INTEL                   0x4109
#define CL_KERNEL_COMPILE_SUB_GROUP_SIZE_INTEL           0x410A

/****************************************
* cl_intel_driver_diagnostics extension *
*****************************************/

#define cl_intel_driver_diagnostics 1

typedef cl_uint cl_diagnostics_verbose_level;

#define CL_CONTEXT_SHOW_DIAGNOSTICS_INTEL                0x4106

#define CL_CONTEXT_DIAGNOSTICS_LEVEL_ALL_INTEL           ( 0xff )
#define CL_CONTEXT_DIAGNOSTICS_LEVEL_GOOD_INTEL          ( 1 )
#define CL_CONTEXT_DIAGNOSTICS_LEVEL_BAD_INTEL           ( 1 << 1 )
#define CL_CONTEXT_DIAGNOSTICS_LEVEL_NEUTRAL_INTEL       ( 1 << 2 )

/********************************
* cl_intel_planar_yuv extension *
*********************************/

#define CL_NV12_INTEL                                       0x410E

#define CL_MEM_NO_ACCESS_INTEL                              ( 1 << 24 )
#define CL_MEM_ACCESS_FLAGS_UNRESTRICTED_INTEL              ( 1 << 25 )

#define CL_DEVICE_PLANAR_YUV_MAX_WIDTH_INTEL                0x417E
#define CL_DEVICE_PLANAR_YUV_MAX_HEIGHT_INTEL               0x417F

/*******************************************************
* cl_intel_device_side_avc_motion_estimation extension *
********************************************************/

#define CL_DEVICE_AVC_ME_VERSION_INTEL                      0x410B
#define CL_DEVICE_AVC_ME_SUPPORTS_TEXTURE_SAMPLER_USE_INTEL 0x410C
#define CL_DEVICE_AVC_ME_SUPPORTS_PREEMPTION_INTEL          0x410D

#define CL_AVC_ME_VERSION_0_INTEL                           0x0   /* No support. */
#define CL_AVC_ME_VERSION_1_INTEL                           0x1   /* First supported version. */

#define CL_AVC_ME_MAJOR_16x16_INTEL                         0x0
#define CL_AVC_ME_MAJOR_16x8_INTEL                          0x1
#define CL_AVC_ME_MAJOR_8x16_INTEL                          0x2
#define CL_AVC_ME_MAJOR_8x8_INTEL                           0x3

#define CL_AVC_ME_MINOR_8x8_INTEL                           0x0
#define CL_AVC_ME_MINOR_8x4_INTEL                           0x1
#define CL_AVC_ME_MINOR_4x8_INTEL                           0x2
#define CL_AVC_ME_MINOR_4x4_INTEL                           0x3

#define CL_AVC_ME_MAJOR_FORWARD_INTEL                       0x0
#define CL_AVC_ME_MAJOR_BACKWARD_INTEL                      0x1
#define CL_AVC_ME_MAJOR_BIDIRECTIONAL_INTEL                 0x2

#define CL_AVC_ME_PARTITION_MASK_ALL_INTEL                  0x0
#define CL_AVC_ME_PARTITION_MASK_16x16_INTEL                0x7E
#define CL_AVC_ME_PARTITION_MASK_16x8_INTEL                 0x7D
#define CL_AVC_ME_PARTITION_MASK_8x16_INTEL                 0x7B
#define CL_AVC_ME_PARTITION_MASK_8x8_INTEL                  0x77
#define CL_AVC_ME_PARTITION_MASK_8x4_INTEL                  0x6F
#define CL_AVC_ME_PARTITION_MASK_4x8_INTEL                  0x5F
#define CL_AVC_ME_PARTITION_MASK_4x4_INTEL                  0x3F

#define CL_AVC_ME_SEARCH_WINDOW_EXHAUSTIVE_INTEL            0x0
#define CL_AVC_ME_SEARCH_WINDOW_SMALL_INTEL                 0x1
#define CL_AVC_ME_SEARCH_WINDOW_TINY_INTEL                  0x2
#define CL_AVC_ME_SEARCH_WINDOW_EXTRA_TINY_INTEL            0x3
#define CL_AVC_ME_SEARCH_WINDOW_DIAMOND_INTEL               0x4
#define CL_AVC_ME_SEARCH_WINDOW_LARGE_DIAMOND_INTEL         0x5
#define CL_AVC_ME_SEARCH_WINDOW_RESERVED0_INTEL             0x6
#define CL_AVC_ME_SEARCH_WINDOW_RESERVED1_INTEL             0x7
#define CL_AVC_ME_SEARCH_WINDOW_CUSTOM_INTEL                0x8
#define CL_AVC_ME_SEARCH_WINDOW_16x12_RADIUS_INTEL          0x9
#define CL_AVC_ME_SEARCH_WINDOW_4x4_RADIUS_INTEL            0x2
#define CL_AVC_ME_SEARCH_WINDOW_2x2_RADIUS_INTEL            0xa

#define CL_AVC_ME_SAD_ADJUST_MODE_NONE_INTEL                0x0
#define CL_AVC_ME_SAD_ADJUST_MODE_HAAR_INTEL                0x2

#define CL_AVC_ME_SUBPIXEL_MODE_INTEGER_INTEL               0x0
#define CL_AVC_ME_SUBPIXEL_MODE_HPEL_INTEL                  0x1
#define CL_AVC_ME_SUBPIXEL_MODE_QPEL_INTEL                  0x3

#define CL_AVC_ME_COST_PRECISION_QPEL_INTEL                 0x0
#define CL_AVC_ME_COST_PRECISION_HPEL_INTEL                 0x1
#define CL_AVC_ME_COST_PRECISION_PEL_INTEL                  0x2
#define CL_AVC_ME_COST_PRECISION_DPEL_INTEL                 0x3

#define CL_AVC_ME_BIDIR_WEIGHT_QUARTER_INTEL                0x10
#define CL_AVC_ME_BIDIR_WEIGHT_THIRD_INTEL                  0x15
#define CL_AVC_ME_BIDIR_WEIGHT_HALF_INTEL                   0x20
#define CL_AVC_ME_BIDIR_WEIGHT_TWO_THIRD_INTEL              0x2B
#define CL_AVC_ME_BIDIR_WEIGHT_THREE_QUARTER_INTEL          0x30

#define CL_AVC_ME_BORDER_REACHED_LEFT_INTEL                 0x0
#define CL_AVC_ME_BORDER_REACHED_RIGHT_INTEL                0x2
#define CL_AVC_ME_BORDER_REACHED_TOP_INTEL                  0x4
#define CL_AVC_ME_BORDER_REACHED_BOTTOM_INTEL               0x8

#define CL_AVC_ME_SKIP_BLOCK_PARTITION_16x16_INTEL          0x0
#define CL_AVC_ME_SKIP_BLOCK_PARTITION_8x8_INTEL            0x4000

#define CL_AVC_ME_SKIP_BLOCK_16x16_FORWARD_ENABLE_INTEL     ( 0x1 << 24 )
#define CL_AVC_ME_SKIP_BLOCK_16x16_BACKWARD_ENABLE_INTEL    ( 0x2 << 24 )
#define CL_AVC_ME_SKIP_BLOCK_16x16_DUAL_ENABLE_INTEL        ( 0x3 << 24 )
#define CL_AVC_ME_SKIP_BLOCK_8x8_FORWARD_ENABLE_INTEL       ( 0x55 << 24 )
#define CL_AVC_ME_SKIP_BLOCK_8x8_BACKWARD_ENABLE_INTEL      ( 0xAA << 24 )
#define CL_AVC_ME_SKIP_BLOCK_8x8_DUAL_ENABLE_INTEL          ( 0xFF << 24 )
#define CL_AVC_ME_SKIP_BLOCK_8x8_0_FORWARD_ENABLE_INTEL     ( 0x1 << 24 )
#define CL_AVC_ME_SKIP_BLOCK_8x8_0_BACKWARD_ENABLE_INTEL    ( 0x2 << 24 )
#define CL_AVC_ME_SKIP_BLOCK_8x8_1_FORWARD_ENABLE_INTEL     ( 0x1 << 26 )
#define CL_AVC_ME_SKIP_BLOCK_8x8_1_BACKWARD_ENABLE_INTEL    ( 0x2 << 26 )
#define CL_AVC_ME_SKIP_BLOCK_8x8_2_FORWARD_ENABLE_INTEL     ( 0x1 << 28 )
#define CL_AVC_ME_SKIP_BLOCK_8x8_2_BACKWARD_ENABLE_INTEL    ( 0x2 << 28 )
#define CL_AVC_ME_SKIP_BLOCK_8x8_3_FORWARD_ENABLE_INTEL     ( 0x1 << 30 )
#define CL_AVC_ME_SKIP_BLOCK_8x8_3_BACKWARD_ENABLE_INTEL    ( 0x2 << 30 )

#define CL_AVC_ME_BLOCK_BASED_SKIP_4x4_INTEL                0x00
#define CL_AVC_ME_BLOCK_BASED_SKIP_8x8_INTEL                0x80

#define CL_AVC_ME_INTRA_16x16_INTEL                         0x0
#define CL_AVC_ME_INTRA_8x8_INTEL                           0x1
#define CL_AVC_ME_INTRA_4x4_INTEL                           0x2

#define CL_AVC_ME_INTRA_LUMA_PARTITION_MASK_16x16_INTEL     0x6
#define CL_AVC_ME_INTRA_LUMA_PARTITION_MASK_8x8_INTEL       0x5
#define CL_AVC_ME_INTRA_LUMA_PARTITION_MASK_4x4_INTEL       0x3

#define CL_AVC_ME_INTRA_NEIGHBOR_LEFT_MASK_ENABLE_INTEL         0x60
#define CL_AVC_ME_INTRA_NEIGHBOR_UPPER_MASK_ENABLE_INTEL        0x10
#define CL_AVC_ME_INTRA_NEIGHBOR_UPPER_RIGHT_MASK_ENABLE_INTEL  0x8
#define CL_AVC_ME_INTRA_NEIGHBOR_UPPER_LEFT_MASK_ENABLE_INTEL   0x4

#define CL_AVC_ME_LUMA_PREDICTOR_MODE_VERTICAL_INTEL            0x0
#define CL_AVC_ME_LUMA_PREDICTOR_MODE_HORIZONTAL_INTEL          0x1
#define CL_AVC_ME_LUMA_PREDICTOR_MODE_DC_INTEL                  0x2
#define CL_AVC_ME_LUMA_PREDICTOR_MODE_DIAGONAL_DOWN_LEFT_INTEL  0x3
#define CL_AVC_ME_LUMA_PREDICTOR_MODE_DIAGONAL_DOWN_RIGHT_INTEL 0x4
#define CL_AVC_ME_LUMA_PREDICTOR_MODE_PLANE_INTEL               0x4
#define CL_AVC_ME_LUMA_PREDICTOR_MODE_VERTICAL_RIGHT_INTEL      0x5
#define CL_AVC_ME_LUMA_PREDICTOR_MODE_HORIZONTAL_DOWN_INTEL     0x6
#define CL_AVC_ME_LUMA_PREDICTOR_MODE_VERTICAL_LEFT_INTEL       0x7
#define CL_AVC_ME_LUMA_PREDICTOR_MODE_HORIZONTAL_UP_INTEL       0x8
#define CL_AVC_ME_CHROMA_PREDICTOR_MODE_DC_INTEL                0x0
#define CL_AVC_ME_CHROMA_PREDICTOR_MODE_HORIZONTAL_INTEL        0x1
#define CL_AVC_ME_CHROMA_PREDICTOR_MODE_VERTICAL_INTEL          0x2
#define CL_AVC_ME_CHROMA_PREDICTOR_MODE_PLANE_INTEL             0x3

#define CL_AVC_ME_FRAME_FORWARD_INTEL                       0x1
#define CL_AVC_ME_FRAME_BACKWARD_INTEL                      0x2
#define CL_AVC_ME_FRAME_DUAL_INTEL                          0x3

#define CL_AVC_ME_SLICE_TYPE_PRED_INTEL                     0x0
#define CL_AVC_ME_SLICE_TYPE_BPRED_INTEL                    0x1
#define CL_AVC_ME_SLICE_TYPE_INTRA_INTEL                    0x2

#define CL_AVC_ME_INTERLACED_SCAN_TOP_FIELD_INTEL           0x0
#define CL_AVC_ME_INTERLACED_SCAN_BOTTOM_FIELD_INTEL        0x1

/*******************************************
* cl_intel_unified_shared_memory extension *
********************************************/

/* These APIs are in sync with Revision Q of the cl_intel_unified_shared_memory spec! */

#define cl_intel_unified_shared_memory 1

/* cl_device_info */
#define CL_DEVICE_HOST_MEM_CAPABILITIES_INTEL                   0x4190
#define CL_DEVICE_DEVICE_MEM_CAPABILITIES_INTEL                 0x4191
#define CL_DEVICE_SINGLE_DEVICE_SHARED_MEM_CAPABILITIES_INTEL   0x4192
#define CL_DEVICE_CROSS_DEVICE_SHARED_MEM_CAPABILITIES_INTEL    0x4193
#define CL_DEVICE_SHARED_SYSTEM_MEM_CAPABILITIES_INTEL          0x4194

typedef cl_bitfield cl_device_unified_shared_memory_capabilities_intel;

/* cl_device_unified_shared_memory_capabilities_intel - bitfield */
#define CL_UNIFIED_SHARED_MEMORY_ACCESS_INTEL                   (1 << 0)
#define CL_UNIFIED_SHARED_MEMORY_ATOMIC_ACCESS_INTEL            (1 << 1)
#define CL_UNIFIED_SHARED_MEMORY_CONCURRENT_ACCESS_INTEL        (1 << 2)
#define CL_UNIFIED_SHARED_MEMORY_CONCURRENT_ATOMIC_ACCESS_INTEL (1 << 3)

typedef cl_properties cl_mem_properties_intel;

/* cl_mem_properties_intel */
#define CL_MEM_ALLOC_FLAGS_INTEL        0x4195

typedef cl_bitfield cl_mem_alloc_flags_intel;

/* cl_mem_alloc_flags_intel - bitfield */
#define CL_MEM_ALLOC_WRITE_COMBINED_INTEL               (1 << 0)

typedef cl_uint cl_mem_info_intel;

/* cl_mem_alloc_info_intel */
#define CL_MEM_ALLOC_TYPE_INTEL         0x419A
#define CL_MEM_ALLOC_BASE_PTR_INTEL     0x419B
#define CL_MEM_ALLOC_SIZE_INTEL         0x419C
#define CL_MEM_ALLOC_DEVICE_INTEL       0x419D
/* Enum values 0x419E-0x419F are reserved for future queries. */

typedef cl_uint cl_unified_shared_memory_type_intel;

/* cl_unified_shared_memory_type_intel */
#define CL_MEM_TYPE_UNKNOWN_INTEL       0x4196
#define CL_MEM_TYPE_HOST_INTEL          0x4197
#define CL_MEM_TYPE_DEVICE_INTEL        0x4198
#define CL_MEM_TYPE_SHARED_INTEL        0x4199

typedef cl_uint cl_mem_advice_intel;

/* cl_mem_advice_intel */
/* Enum values 0x4208-0x420F are reserved for future memory advices. */

/* cl_kernel_exec_info */
#define CL_KERNEL_EXEC_INFO_INDIRECT_HOST_ACCESS_INTEL      0x4200
#define CL_KERNEL_EXEC_INFO_INDIRECT_DEVICE_ACCESS_INTEL    0x4201
#define CL_KERNEL_EXEC_INFO_INDIRECT_SHARED_ACCESS_INTEL    0x4202
#define CL_KERNEL_EXEC_INFO_USM_PTRS_INTEL                  0x4203

/* cl_command_type */
#define CL_COMMAND_MEMFILL_INTEL        0x4204
#define CL_COMMAND_MEMCPY_INTEL         0x4205
#define CL_COMMAND_MIGRATEMEM_INTEL     0x4206
#define CL_COMMAND_MEMADVISE_INTEL      0x4207

extern CL_API_ENTRY void* CL_API_CALL
clHostMemAllocINTEL(
            cl_context context,
            const cl_mem_properties_intel* properties,
            size_t size,
            cl_uint alignment,
            cl_int* errcode_ret);

typedef CL_API_ENTRY void* (CL_API_CALL *
clHostMemAllocINTEL_fn)(
            cl_context context,
            const cl_mem_properties_intel* properties,
            size_t size,
            cl_uint alignment,
            cl_int* errcode_ret);

extern CL_API_ENTRY void* CL_API_CALL
clDeviceMemAllocINTEL(
            cl_context context,
            cl_device_id device,
            const cl_mem_properties_intel* properties,
            size_t size,
            cl_uint alignment,
            cl_int* errcode_ret);

typedef CL_API_ENTRY void* (CL_API_CALL *
clDeviceMemAllocINTEL_fn)(
            cl_context context,
            cl_device_id device,
            const cl_mem_properties_intel* properties,
            size_t size,
            cl_uint alignment,
            cl_int* errcode_ret);

extern CL_API_ENTRY void* CL_API_CALL
clSharedMemAllocINTEL(
            cl_context context,
            cl_device_id device,
            const cl_mem_properties_intel* properties,
            size_t size,
            cl_uint alignment,
            cl_int* errcode_ret);

typedef CL_API_ENTRY void* (CL_API_CALL *
clSharedMemAllocINTEL_fn)(
            cl_context context,
            cl_device_id device,
            const cl_mem_properties_intel* properties,
            size_t size,
            cl_uint alignment,
            cl_int* errcode_ret);

extern CL_API_ENTRY cl_int CL_API_CALL
clMemFreeINTEL(
            cl_context context,
            void* ptr);

typedef CL_API_ENTRY cl_int (CL_API_CALL *
clMemFreeINTEL_fn)(
            cl_context context,
            void* ptr);

extern CL_API_ENTRY cl_int CL_API_CALL
clMemBlockingFreeINTEL(
            cl_context context,
            void* ptr);

typedef CL_API_ENTRY cl_int (CL_API_CALL *
clMemBlockingFreeINTEL_fn)(
            cl_context context,
            void* ptr);

extern CL_API_ENTRY cl_int CL_API_CALL
clGetMemAllocInfoINTEL(
            cl_context context,
            const void* ptr,
            cl_mem_info_intel param_name,
            size_t param_value_size,
            void* param_value,
            size_t* param_value_size_ret);

typedef CL_API_ENTRY cl_int (CL_API_CALL *
clGetMemAllocInfoINTEL_fn)(
            cl_context context,
            const void* ptr,
            cl_mem_info_intel param_name,
            size_t param_value_size,
            void* param_value,
            size_t* param_value_size_ret);

extern CL_API_ENTRY cl_int CL_API_CALL
clSetKernelArgMemPointerINTEL(
            cl_kernel kernel,
            cl_uint arg_index,
            const void* arg_value);

typedef CL_API_ENTRY cl_int (CL_API_CALL *
clSetKernelArgMemPointerINTEL_fn)(
            cl_kernel kernel,
            cl_uint arg_index,
            const void* arg_value);

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueMemsetINTEL(       /* Deprecated */
            cl_command_queue command_queue,
            void* dst_ptr,
            cl_int value,
            size_t size,
            cl_uint num_events_in_wait_list,
            const cl_event* event_wait_list,
            cl_event* event);

typedef CL_API_ENTRY cl_int (CL_API_CALL *
clEnqueueMemsetINTEL_fn)(   /* Deprecated */
            cl_command_queue command_queue,
            void* dst_ptr,
            cl_int value,
            size_t size,
            cl_uint num_events_in_wait_list,
            const cl_event* event_wait_list,
            cl_event* event);

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueMemFillINTEL(
            cl_command_queue command_queue,
            void* dst_ptr,
            const void* pattern,
            size_t pattern_size,
            size_t size,
            cl_uint num_events_in_wait_list,
            const cl_event* event_wait_list,
            cl_event* event);

typedef CL_API_ENTRY cl_int (CL_API_CALL *
clEnqueueMemFillINTEL_fn)(
            cl_command_queue command_queue,
            void* dst_ptr,
            const void* pattern,
            size_t pattern_size,
            size_t size,
            cl_uint num_events_in_wait_list,
            const cl_event* event_wait_list,
            cl_event* event);

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueMemcpyINTEL(
            cl_command_queue command_queue,
            cl_bool blocking,
            void* dst_ptr,
            const void* src_ptr,
            size_t size,
            cl_uint num_events_in_wait_list,
            const cl_event* event_wait_list,
            cl_event* event);

typedef CL_API_ENTRY cl_int (CL_API_CALL *
clEnqueueMemcpyINTEL_fn)(
            cl_command_queue command_queue,
            cl_bool blocking,
            void* dst_ptr,
            const void* src_ptr,
            size_t size,
            cl_uint num_events_in_wait_list,
            const cl_event* event_wait_list,
            cl_event* event);

#ifdef CL_VERSION_1_2

/* Because these APIs use cl_mem_migration_flags, they require
   OpenCL 1.2: */

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueMigrateMemINTEL(
            cl_command_queue command_queue,
            const void* ptr,
            size_t size,
            cl_mem_migration_flags flags,
            cl_uint num_events_in_wait_list,
            const cl_event* event_wait_list,
            cl_event* event);

typedef CL_API_ENTRY cl_int (CL_API_CALL *
clEnqueueMigrateMemINTEL_fn)(
            cl_command_queue command_queue,
            const void* ptr,
            size_t size,
            cl_mem_migration_flags flags,
            cl_uint num_events_in_wait_list,
            const cl_event* event_wait_list,
            cl_event* event);

#endif

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueMemAdviseINTEL(
            cl_command_queue command_queue,
            const void* ptr,
            size_t size,
            cl_mem_advice_intel advice,
            cl_uint num_events_in_wait_list,
            const cl_event* event_wait_list,
            cl_event* event);

typedef CL_API_ENTRY cl_int (CL_API_CALL *
clEnqueueMemAdviseINTEL_fn)(
            cl_command_queue command_queue,
            const void* ptr,
            size_t size,
            cl_mem_advice_intel advice,
            cl_uint num_events_in_wait_list,
            const cl_event* event_wait_list,
            cl_event* event);

/***************************************************
* cl_intel_create_buffer_with_properties extension *
****************************************************/

#define cl_intel_create_buffer_with_properties 1

extern CL_API_ENTRY cl_mem CL_API_CALL
clCreateBufferWithPropertiesINTEL(
    cl_context   context,
    const cl_mem_properties_intel* properties,
    cl_mem_flags flags,
    size_t       size,
    void *       host_ptr,
    cl_int *     errcode_ret) CL_EXT_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_mem (CL_API_CALL *
clCreateBufferWithPropertiesINTEL_fn)(
    cl_context   context,
    const cl_mem_properties_intel* properties,
    cl_mem_flags flags,
    size_t       size,
    void *       host_ptr,
    cl_int *     errcode_ret) CL_EXT_SUFFIX__VERSION_1_0;

/******************************************
* cl_intel_mem_channel_property extension *
*******************************************/

#define CL_MEM_CHANNEL_INTEL            0x4213

/*********************************
* cl_intel_mem_force_host_memory *
**********************************/

#define cl_intel_mem_force_host_memory 1

/* cl_mem_flags */
#define CL_MEM_FORCE_HOST_MEMORY_INTEL                      (1 << 20)

#ifdef __cplusplus
}
#endif


#endif /* __CL_EXT_H */
