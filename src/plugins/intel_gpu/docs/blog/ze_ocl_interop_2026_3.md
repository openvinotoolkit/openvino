# Support for Level Zero - OpenCL interoperability: New in OpenVINO 2026.3

*By Jakub Kasprzak | June 23, 2026*

Some projects maintain their own OpenCL GPU pipeline and require features such as memory, context or queue sharing. For such users OpenVINO GPU plugin implements Remote Context and Remote Tensor interface. However, this OpenCL interoperability works only if OpenVINO GPU runtime is set to OpenCL and prevents OpenVINO from switching to Level Zero runtime.
OpenVINO 2026.3 introduces Level Zero - OpenCL interoperability enabling users that depend on OpenVINO Remote Context/Tensor OpenCL API to swap OpenVINO GPU runtime to Level Zero without changing their OpenCL based integration logic. This post explains: how to enable this feature and provides OpenVINO maintainers with design description.

## How to enable Level Zero - OpenCL interoperability

OpenVINO GPU plugin with Level Zero runtime detects if interoperability is supported on current system and enables it automatically. User must ensure that currently installed Intel GPU driver supports and enables "Level Zero executing OpenCL" (LEO) feature.

## Level Zero - OpenCL interoperability design

*This section describes internal design of the feature and is intended for OpenVINO maintainers*

### Object lifetimes

This subsection describes part of the design responsible for managing resources. Main goal was to provide modular API that will minimize chances of memory leaks and other memory bugs.

Logic for managing all (Level Zero and OpenCL) GPU resources is provided by `resource_owner` template class. It contains `is_borrowed` flag to distinguish between resources that need to be released and resources that should not be released (borrowed resources, for example: external memory buffers provided by the user). Once `resource_owner` is destroyed it will release the resource or do nothing if resource was borrowed. Basically `resource_owner` with `is_borrowed=false` is a generic RAII wrapper which means that only one such object can exist for a given resource and that it can not be copied (only move semantics are allowed).

In order to work correctly `resource_owner` template must be provided with information about the type of resource it manages and how to release it. This information is defined for Level Zero and OpenCL resources in `ze_resource_info` and `ocl_resource_info` class template specializations respectively. For example `ocl_resource_info<ocl_resource_type::command_queue>` defines `handle_t` as `cl_command_queue` and `deleter_t` as a functor that calls `clReleaseCommandQueue`. Finally `ze_owner` and `ocl_owner` aliases are defined by combining `resource_owner` with `ze_resource_info` and `ocl_resource_info` respectively.

### Binding corresponding Level Zero - OpenCL handles

It is important to recognize that with interoperability some Level Zero handles will have corresponding OpenCL handles and vice versa. For example, after converting Level Zero immediate command list to OpenCL command queue we end up with two different handles referring to the same entity. `ze_ocl_owner_impl` is a template that is responsible for combining Level Zero handle with OpenCL handle and managing them as a single resource. Internally it contains `ze_owner` and tuple of optional `ocl_owner`s. By design `ze_ocl_owner_impl` always contains Level Zero resource and OpenCL resources can be attached to it. Once OpenCL resource of some given type is attached to `ze_ocl_owner_impl` attempt to attach another OpenCL resource of the same type will result in **exception** (this design is to prevent exporting single Level Zero resource multiple times).

Not all handle combinations are valid. For example in current context it does not make sense to attach/bind `cl_command_queue` handle to `ze_ocl_owner_impl` that holds `ze_image_handle_t`. To prevent this `ze_ocl_owner` template inherits only valid `ze_ocl_owner_impl` specializations. For example `ze_ocl_owner<ze_resource_type::command_list>` is `ze_ocl_owner_impl<ze_resource_type::command_list, ocl_resource_type::command_queue>` meaning it holds `ze_command_list_handle_t` and optionally `cl_command_queue` handles. If specialization is not defined explicitly it fall backs to default implementation where only Level Zero handle is allowed, for example `ze_ocl_owner<ze_resource_type::kernel>` holds only `ze_kernel_handle_t` and attempt to attach or access any type of OpenCL handle will result in **compilation error**.

### Shared resource ownership

Finally `ze_resource` template combines `std::shared_ptr` and `ze_ocl_owner` to implement shared ownership model for GPU resources. This is especially useful for handling lifetime of Level Zero events and event pools where `ze_resource<ze_resource_type::event_pool>` can be copied to all event objects created from given pool ensuring that the pool resource will be destroyed only after last event from the pool was destroyed (same scheme is used to manage lifetime of Level Zero modules and kernels).

Additionaly aliases in form of `ze_*_resource` such as `ze_event_pool_resource` are defined for developer convenience.

### Interoperability integration

Basic interoperability is implemented by a `ze_ocl_interop` singleton that detects interoperability support and provides direct (no abstraction) Level Zero - OpenCL handle to handle conversion interface. When `ze_ocl_interop` is accessed for the first time it will initialize GPU runtimes and map corresponding Level Zero and OpenCL devices.

Conversion interface provided by `ze_ocl_interop` is wrapped by `ze_import_*` and `ze_export_ocl` functions from `ze_resource_interop.hpp` in order to simplify conversion logic when used with `ze_resource` objects. For example calling `ze_export_ocl_command_queue` for given `ze_command_list_resource` will:
1. Return early if `ze_command_list_resource` already has attached OpenCL command queue handle
2. Create new OpenCL command queue based on present Level Zero handle
3. Attach new OpenCL handle to `ze_command_list_resource`
4. Guarantee strong exception safety

### Remote context

Type of remote context decides what kind of objects are exposed by the GPU plugin. During model compilation OpenVINO creates remote context and internal engine that by default match currently selected GPU runtime. For example, when GPU runtime is set to Level Zero this causes the creation of Level Zero remote context and `ze_engine`. Remote context will query internal GPU plugin objects for handles based on the remote context type. So if Level Zero remote context is used only Level Zero handles can be requested from internal objects and OpenCL interoperability is not required.

If user provides OpenCL context or queue or OpenVINO detects that current system supports Level Zero - OpenCL interoperability then remote context type will be changed to OpenCL but engine will still match currently selected GPU runtime. In result OpenCL remote context will request OpenCL handles from internal Level Zero GPU plugin objects like `ze_engine` forcing usage of interoperability features.
