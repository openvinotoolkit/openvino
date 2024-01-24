# In memory cache

## Motivation

When creating a primitive_impl in the Dynamic Shape model, if each primitive_impls are created about the same primitive with the same type and input / output shapes, it creates duplicated primitive_impl including new cl kernel build for same kernel source. this may result in inefficiency and performance degradation due to build the exactly same cl kernel source code multiple times for same layout and primitive type on the run time for dynamic model. To resolve this issue, ImplementationCache handle is newly introduced.


## Property

* ImplementationCache only handles primitive_impl which is created in primitive_inst::update_impl() and primitive_inst::update_weights() on dynamic shape model. In the case of static shape, kernels_cache handles static shape kernel duplication.
* ImplementationCache inherits LRUCacheThreadSafe which is ThreadSafe version of LRUCache which handles primitive_impl cache by increasing the cache hit rate for frequently used items. Therefore, ImplementationCache optimizes the performance of dynamic execution through frequently used primitive_impl.
* Since cldnn::program creates ImplementationCache as unique_ptr at cldnn::program constructor, its lifecycle is set to cldnn::program.
* ImplementationCache supports multi-stream, so the cldnn::network of each stream manages primitive_impl in same cache.
* ImplementationCache Capacity is set to 10000 by default, but may change in the future optimization.


## Usages

ImplementationCache is used to handle primitive_impl cache at primitive_inst::update_impl() and primitive_inst::update_weights() in dynamic shape model.

* In primitive_inst::update_impl(), it looks up the cache with key which is hash value of kernel_impl_param which is updated by the current primitive_inst. If it is not found from ImplementationCache, new primitive_impl is created and save it into the cache.
* In primitive_inst::update_weights(), if it is not found a primitive_impl with a hash key value which matches the weights_reorder_kernel_params of the primitive inst, it also create a new primitive_impl for weight reorder and put it in the cache.
