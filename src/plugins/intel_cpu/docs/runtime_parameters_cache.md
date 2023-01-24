# CPU plugin runtime parameters cache

## Checklist for the runtime cache implementation
1. Determine what data will be cached. We usually use the Executor concept that represents a junction of the executable code, usually JIT generated kernel, with some precomputed algorithm parameters.
2. Provide a key that uniquelly identifies the cached value as a funtion of dynamically changing parameters, i.e. shapes, dynamic input that determines the algorithm parameters, etc. To be used in a hash table, the key must have the following static interface:
   ```cpp
   struct KeyType {
       size_t hash() const;
       bool operator== () const;
   };
   ```
3. Provide a builder, that is, a callable object of the following signature: 
   ```cpp
   ValueType build(const KeyType& key);
   ```
   The `ValueType` is a type to be cached (e.g. shared pointer to Executor object). Remember that in the current cache implementation, a default constructed `ValueType()` object is considered empty, so it is better to use `std::shared_ptr` as the `ValueType`. The builder instantiates a specific type of cached entity from the `key`, thus the `key` completely defines the cached data. The builder is used to creat the `ValueType` object in case of cache miss.
4. Refactor the specific implementation of the `prepareParams()` method to extract the cached object construction logic (e.g. the algorithm parameters recalculation and JIT kernel generation) into the builder.
5. Add the key generation code into the `prepareParams()` method to query the cache.
6. Implement cache usage as the following:
   ```cpp
   void preapareParams() override {
        ... //code that prepares parameters for the key

        //key instantiation
        KeyType key = {param1, param2, ...};
        // get a reference to the cache
        auto cache = getRuntimeCache();
        //query cahce, buildExecutor is the builder descibed in 3
        auto result = cache->getOrCreate(key, buildExecutor); 
        // get the the cached value, in this example it is a pointer to an executor
        execPtr = result.first; 
   }
   ```
7. To provide smoke testing of these changes, add repeated shapes to the "target shapes" part of the corresponding single layer test definition:
    ```cpp
    { //dynamic case description each pair per each input has {{dynamic shape}, {{static shape case1}, {static shape case2}, ...}
        {{-1, -1, -1}, {{10, 10, 10}, {5, 5, 5}, {10, 10, 10}}}, // input 0
        {{-1, -1, 5}, {{10, 10, 5}, {5, 5, 5}, {10, 10, 5}}}  // input 1
    },
    ```
   It worth to mention that placing two identical target shapes one after another does not trigger the cache, since another optimization based on the fact that the shapes have not been changed takes place. For example, the following test definition does not properly test the cache:
    ```cpp
    { // the shape infer and params preparation stages will be skipped for the second target shapes combination since the shapes are not changed
        {{-1, -1, -1}, {{5, 5, 5}, {5, 5, 5}}}, // input 0
        {{-1, -1, 5},  {{5, 5, 5}, {5, 5, 5}}}  // input 1
    },
    ```

