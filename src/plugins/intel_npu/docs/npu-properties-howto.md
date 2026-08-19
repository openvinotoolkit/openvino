# NPU Properties
What is a property from NPU Plugin's POV and Step-by-step guide on how to add one  
Practical manual for NPU plugin properties

## Table of Contents

- [Glossary](#glossary)
- [Structure of a property (Class hierarchy)](#structure-of-a-property-class-hierarchy)
    - [Properties With Option vs Properties Without Option](#properties-with-option-vs-properties-without-option)
  - [OptionBase<T>](#optionbaset)
  - [OptionDesc](#optiondesc)
  - [Config](#config)
  - [FilteredConfig](#filteredconfig)
  - [Properties](#properties)
- [How to add a new public (option backed) property](#how-to-add-a-new-public-option-backed-property)
  - [Step 1. Define the new property](#step-1-define-the-new-property)
  - [Step 2. Define the internal option descriptor](#step-2-define-the-internal-option-descriptor)
  - [Step 3. Register the new option](#step-3-register-the-new-option)
  - [Step 4. Link the new property to the new option](#step-4-link-the-new-property-to-the-new-option)
    - [For plugin](#for-plugin)
    - [For compiled-model (if required)](#for-compiled-model-if-required)
  - [Step 5. Python bindings](#step-5-python-bindings)
  - [Step 6. Update documentation](#step-6-update-documentation)
- [How to add a new public property without option (callback-backed/read-only)](#how-to-add-a-new-public-property-without-option-callback-backedread-only)
  - [Step 1. Define the new property](#step-1-define-the-new-property-1)
    - [Step 2. Define and register a callback function for the new property](#step-2-define-and-register-a-callback-function-for-the-new-property)
  - [Step 3. Python bindings](#step-3-python-bindings)
  - [Step 4. Update documentation](#step-4-update-documentation)
- [Compiled-model properties](#compiled-model-properties)
- [Special cases](#special-cases)
  - [SC.1 Adding a new property which requires custom functions](#sc1-adding-a-new-property-which-requires-custom-functions)
    - [SC.2 Adding a new property without option which requires customization](#sc2-adding-a-new-property-without-option-which-requires-customization)
  - [SC.3 Filtering out options at registration phase](#sc3-filtering-out-options-at-registration-phase)
    - [SC.4 Filter compiler configuration before serialization](#sc4-filter-compiler-configuration-before-serialization)
- [Removing a public property](#removing-a-public-property)

## Glossary

| Name     | Description | Example |
|:--------:|:---------   |:--------: |
| Property | a plugin interface which can be set or read | `ov::log::level` | 
| Option   | a configuration entry in our internal configuration.</br>Consists of an OptionBase template descriptor + a template OptionValue value. | `LOG_LEVEL` |
| Config   | Our internal database of configuration keys and their values | |
| Option-backed property | A property mapped to an Option entry in Config (`register_property*` helpers with config argument). | `ov::hint::performance_mode` |
| Property without option (callback-backed/read-only) | A property that is not mapped to Config and is implemented by callback logic (usually runtime/backend queries). | `ov::device::pci_info` |
| Compiler | Npu compiler as viewed from the plugin's perspective.</br>Can be Compiler-In-Driver or Compiler-In-Plugin | `CID` |
| "Anonymous" property</br>OR</br>compiler-private property | A setting from application level which the plugin has no knowledge of</br>(it is not registered, plugin is not aware of its datatype)</br> but which the compiler reports as supported via is_supported() API. | N/A |

<br>

## Structure of a property (Class hierarchy)

![Properties Class Hierarchy](./img/properties_class_hierarchy.png)

<br>

### Properties With Option vs Properties Without Option
As it can be observed in the above class hierarchy diagram, a Property is a public interface to an internal information, the top layer of abstraction.  
A property can be implemented either through Config options or through callback logic.
The main difference is that while option-backed properties are entries in our internal database (Config) which can be modified at any time, properties without option
do not exist in the internal database and are computed/read on demand from backend/driver/OS.
Examples in NPU Plugin are: driver version, compiler version, device architecture, pci information, gops, uuid, luid, etc.
To summarize:
A property can be implemented in **one** of the 2 ways:
- Option-backed property: has an entry in internal Config and is managed through Option descriptors/helpers.
- Property without option: implemented through callback logic (typically read-only runtime/backend information).

### OptionBase\<T\> 
Implements the option descriptor. This class contains all the details of a config option: name, datatype, default value, parser, public/private, mutability, compiler version (for legacy support), etc. This serves as the key in our configuration map. 

Class definition in npu_plugin/al/include/config/config.hpp:  
```cpp
struct OptionBase { 
    using ValueType = T; 

    // `ActualOpt` must implement the following method: 
    // static std::string_view key() 

    static constexpr std::string_view getTypeName() { 
        if constexpr (TypePrinter<T>::hasName()) { 
            return TypePrinter<T>::name(); 
        } 
        static_assert(TypePrinter<T>::hasName(), 
                      "Options type is not a standard type, please add `getTypeName()` to your option"); 
    } 
    // Overload this to provide environment variable support. 
    static std::string_view envVar() { 
        return ""; 
    } 

    // Overload this to provide deprecated keys names. 
    static std::vector<std::string_view> deprecatedKeys() { 
        return {}; 
    } 

    // Overload this to provide default value if it wasn't specified by user. 
    // If it is std::nullopt - exception will be thrown in case of missing option access. 
    static std::optional<T> defaultValue() { 
        return std::nullopt; 
    } 

    // Overload this to provide more specific parser. 
    static ValueType parse(std::string_view val) { 
        return OptionParser<ValueType>::parse(val); 
    } 

    // Overload this to provide more specific validation 
    static void validateValue(const ValueType&) {} 

    // Overload this to provide more specific implementation. 
    static OptionMode mode() { 
        return OptionMode::Both; 
    } 

    // Overload this for private options. 
    static bool isPublic() { 
        return false; 
    } 

    // Overload this for read-only properties 
    static ov::PropertyMutability mutability() { 
        return ov::PropertyMutability::RW; 
    } 

    static std::string toString(const ValueType& val) { 
        return OptionPrinter<ValueType>::toString(val); 
    } 
}; 
```

### OptionsDesc
is storage for the registered options. This is the base map which stores the available OptionBase descriptors.
This layer implements the option database manipulation functions: add/has/reset.
The plugin property manager creates it once, registers the plugin options, and lets the backend add its compiler-specific options.

### Config
is the highlevel configuration "database" which implements the mapping between OptionBase and templatized OptionValue.
Maps and stores the user-defined values for each entry in OptionsDesc layer.
Implements the top level configuration manipulation functions:
get/update/has/getString/toString/fromString and handles typecasts, typeverification, parsing and conversions.
```` Note: This layer is initialized once in the plugin property manager from the finalized OptionsDesc. ````

### FilteredConfig
is a derivative class of Config, used only by NPU Plugin, which implements additional filtering layers atop of the base config,
such as enabling/disabling keys based on their availability/support on the current system configuration.
```` Note: This layer dynamically changes based on system configuration and compiler_type. ````

The initialization order is:
1. `Plugin` creates a small temporary `OptionsDesc`/`FilteredConfig` containing only `LOG_LEVEL` so the logger can be initialized from the environment.
2. `PluginPropertyManager` creates the complete `OptionsDesc` and registers all plugin options.
3. The backend adds its options through `backend->registerOptions(*options)`.
4. The manager constructs its `FilteredConfig` from the finalized descriptor and parses environment variables.
5. The manager resolves the effective compiler type and registers property descriptors.

### Properties
is the top level class and serves as the NPU Plugin's interface to OpenVino and the application layer.
It's main purpose is to implement get_property and set_property interfaces and the callback functions of each property.
```` Note: This layer dynamically changes based on system configuration and compiler_type ````

<br><br>

# How to add a new public (option backed) property

The following steps how to add a new simple property which maps to a compiler configuration option.  
_*simple in this context means that it has no special callback function required for it, just set/get_  

## Step 1. Define the new property
First step is to define the new property's name, datatype and string-name in the public header in  
```bash
src/inference/include/openvino/runtime/intel_npu/properties.hpp
```  
Example:  
```cpp
static constexpr ov::Property<ExampleType,ov::PropertyMutability::RW> example_property{"NPU_EXAMPLE_PROPERTY"};
```
Notes:  
- please note the NPU_ prefix in the property's string name. This is mandatory for npu-only private properties 
- mutability is Read-Write
- datatype of the property is enum ExampleType { VAL1, VAL2, VAL3 }  

## Step 2. Define the internal option descriptor
Second step is to define the optionDesc class for this property in  
```bash
src/plugins/intel_npu/src/al/include/intel_npu/config/options.hpp
```  
Example:  
```cpp
// 
// EXAMPLE_PROPERTY 
//  
struct EXAMPLE_PROPERTY final : OptionBase<EXAMPLE_PROPERTY, ov::intel_npu::ExampleType> {  

    static std::string_view key() { 
        return ov::intel_npu::example_property.name();
        } 

    static constexpr std::string_view getTypeName() { 
        return "ov::intel_npu::ExampleType"; 
    } 

    static ov::intel_npu::ExampleType defaultValue() { 
        return ov::intel_npu::ExampleType::VAL3; 
    } 

    static bool isPublic() { 
        return true; 
    } 

    static OptionMode mode() { 
        return OptionMode::Both; 
    } 
     
    static ov::PropertyMutability mutability() { 
        return ov::PropertyMutability::RW; 
    } 
     
    static std::string_view envVar() { 
        return "IE_NPU_EXAMPLE_PROPERTY"; 
    } 

    static ov::intel_npu::ExampleType parse(std::string_view val) { 
        if (val == "VAL1") { 
            return ov::intel_npu::ExampleType::VAL1; 
        } else if (val == "VAL2") { 
            return ov::intel_npu::ExampleType::VAL2; 
        } else if (val == "VAL3") { 
            return ov::intel_npu::ExampleType::VAL3; 
        } 

        OPENVINO_THROW("Value '", val, "'is not a valid EXAMPLE_PROPERTY option"); 
    } 

    static std::string toString(const ov::intel_npu::ExampleType& val) { 
        std::stringstream strStream; 

        strStream << val; 

        return strStream.str(); 
    } 
}; 
```
Notes:  
- key(): needs to return the string name of the property (the NPU_EXAMPLE_PROPERTY defined in the property at step 1)  
- getTypeName: returns the type name as a human-readable string  
- defaultValue: returns the option's default value (if there was no user-defined value set, config.get or get_property(EXAMPLE_PROPERTY) will call this function)
- isPublic: defines whether the option is a **public or a private** one  
- mode: defines the OptionMode of this option. Can be:  
    - CompileTime (for options used ONLY by the compiler)  
    - Runtime (for options only used by plugin and runtime)  
    - Both (for options used by both).  
    **Only options of CompileTime and Both will be sent to compiler at model compilation.**  
- mutability: whether the option is **Read-Write** or **Read-Only**  
- envVar: environment variable (if needed) for this property. The config manager will check if the options have envVar defined. For each option which has envVar, it will look in environment variables and update the option value from there at init.  
- parse: string to custom datatype parser. If the property will be set with a string value, this parser will convert it into the internal datatype.  
- toString: for converting the option value from the custom datatype to string 

**(!!)** None of the member functions are mandatory to be defined.  
If any is missing, the default function will be used for its call, as defined in the OptionsBase class  
(see class **OptionBase** in `src/plugins/intel_npu/src/al/include/intel_npu/config/config.hpp` or Class Hierarchy section above)

## Step 3. Register the new option
Third step is to register the new option in the plugin property manager:
**src/plugins/intel_npu/src/plugin/src/plugin_property_manager.cpp > function register_options(...)**
```cpp
    REGISTER_OPTION(EXAMPLE_PROPERTY);
``` 
Notes:  
The manager registers the option in `OptionsDesc` before constructing its `FilteredConfig`. The backend options are registered immediately afterward, and environment variables are parsed on the completed configuration.

## Step 4. Link the new property to the new option
Fourth step is to create and register the Property (which is basically the interface to this configuration option) for both Plugin and CompiledModel (if needed) 
### For plugin
src/plugins/intel_npu/src/plugin/src/plugin_property_manager.cpp > function PluginPropertyManager::registerProperties()
```cpp
register_property<EXAMPLE_PROPERTY>(_config, _properties, true, ov::PropertyMutability::RW);
```
**Explanation:**
this helper function registers a property with the name ov::intel_npu::example_property.name()  
which maps to our internal configuration named EXAMPLE_PROPERTY, and is supported when the option is available.  
and has a simple callback function of config.get<EXAMPLE_PROPERTY>()
### For compiled-model (if required)
src/plugins/intel_npu/src/plugin/src/compiled_model_property_manager.cpp > function CompiledModelPropertyManager::registerProperties()
```cpp
    register_property_with_support<EXAMPLE_PROPERTY>(_config,
                                                     _properties,
                                                     true,
                                                     ov::PropertyMutability::RO,
                                                     [hasPropertyValue](const ov::AnyMap&) {
                                                         return hasPropertyValue(ov::intel_npu::example_property.name());
                                                     });
```
**Explanation:**
this helper function registers the compiled-model property with the name ov::intel_npu::example_property.name()
which maps to our internal configuration named EXAMPLE_PROPERTY
and has a simple callback function of config.get<EXAMPLE_PROPERTY>()
**(!!) ONLY** if it has been previously explicitly set at compile time.

## Step 5. Python bindings
In order for the property to be exposed in Python API, add python wrapper for the new property in pyOpenvino  
src/bindings/python/src/pyopenvino/core/properties/properties.cpp:  
In section // submodule npu  
```cpp
    wrap_property_RW(m_intel_npu, ov::intel_npu::example_property, "example_property"); 
```

## Step 6. Update documentation
Document the new property in the appropriate sections (+ additional information, if required) in:  
```bash
docs/articles_en/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.rst 
src/plugins/intel_npu/README.md 
```

<br><br>

# How to add a new public property without option (callback-backed/read-only)
As described in the first paragraph, these properties do not have an entry in our internal Config and are returned through callback logic at get_property calls.  
This means we do not need to define an Option nor register an Option for them.  

## Step 1. Define the new property
First step is to define the new property's name, datatype and string-name in the public header in  
```bash
src/inference/include/openvino/runtime/intel_npu/properties.hpp
```  
Example:  
```cpp
static constexpr ov::Property<ExampleType,ov::PropertyMutability::RO> example_property{"NPU_EXAMPLE_PROPERTY"};
```
Notes:  
- please note the NPU_ prefix in the property's string name. This is mandatory for npu-only private properties 
- mutability is Read-Only
- datatype of the property is enum ExampleType { VAL1, VAL2, VAL3 } 

## Step 2. Define and register a callback function for the new property
You need to register the new property and define a callback function in the owner-specific property manager.
For plugin: src/plugins/intel_npu/src/plugin/src/plugin_property_manager.cpp > function PluginPropertyManager::registerProperties()
For compiled-model: src/plugins/intel_npu/src/plugin/src/compiled_model_property_manager.cpp > function CompiledModelPropertyManager::registerProperties()

For properties without option, prefer a support-gated registration so the getter is only used when backend/runtime requirements are available:
```cpp
    const auto has_backend = [this]() {
        return _backend != nullptr;
    };

    register_property_with_support_and_custom_function(_properties,
                                                       ov::intel_npu::example_property.name(),
                                                       true,
                                                       ov::PropertyMutability::RO,
                                                       has_backend,
                                                       [this](const ov::AnyMap&) {
                                                           return utils::getDriverVersion(_backend);
                                                       });
```
**Explanation**
this helper function registers a property with the name **ov::intel_npu::example_property (NPU_EXAMPLE_PROPERTY)**, which is public and included in supported_properties when the support predicate returns true.  
and calls `utils::getDriverVersion(_backend)` each time get_property is queried. The getter receives query-time
arguments as `ov::AnyMap`; use them when the property supports extra inputs, or ignore them when it does not.
Note: the first argument is the property name string (`property.name()`), not the property object itself.

## Step 3. Python bindings
In order for the property to be exposed in Python API, add python wrapper for the new property in pyOpenvino  
src/bindings/python/src/pyopenvino/core/properties/properties.cpp:  
In section // submodule npu  
```cpp
    wrap_property_RO(m_intel_npu, ov::intel_npu::example_property, "example_property"); 
```

## Step 4. Update documentation
Document the new property in the appropriate sections (+ additional information, if required) in:  
```bash
docs/articles_en/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.rst
src/plugins/intel_npu/README.md
```

<br><br>

# Compiled-model properties
By internal convention, what needs to be included in compiled-model properties gets decided based on the following statements:
- every option which has an effect on model compilation (meaning options with mode=OptionMode::CompileTime or OptionMode::Both) need to be included
- options (with some specific exceptions) should be publicly advertised in compiled-model's supported_properties **ONLY** if they have been explicitly set prior to model compilation.
- compiled-model properties (with a few specific exceptions) are all READ-ONLY, for the reason that the model has already been compiled.
This is to ensure that we only expose settings we are sure were taken into account by compiler.

For read-only config-backed properties in compiled-model, use the same registration helpers as the plugin manager and pass
`ov::PropertyMutability::RO`. Use plain `register_property` when the option should be advertised whenever it is registered,
or `register_property_with_support` with a `hasPropertyValue` predicate when it should be advertised only if explicitly set.

#### `register_property_with_support<OPT_TYPE>(config, properties, isPublic, mutability, isSupported)
Use this when availability depends on runtime state. In compiled-model, `hasPropertyValue` checks whether the option was
explicitly set; default values are resolved from `OptionsDesc` and are not materialized in the compiled-model config.
Example:
```cpp
    const auto hasPropertyValue = [this](std::string_view propertyName) {
        return _config.hasOpt(propertyName) && _config.has(std::string(propertyName));
    };

    register_property_with_support<COMPILATION_MODE>(
        _config,
        _properties,
        true,
        ov::PropertyMutability::RO,
        [hasPropertyValue](const ov::AnyMap&) {
            return hasPropertyValue(ov::intel_npu::compilation_mode.name());
        });
```

#### `register_property_with_custom_function` for a config-backed option
Use the overload with `config` and an `OptionType` when the option is registered in `FilteredConfig` but its getter is
custom. Visibility and mutability are supplied by the caller, and support defaults to `config.hasOpt(option key)`.
Example:
```cpp
    register_property_with_custom_function<COMPILE_LOG_LEVEL>(
        _config,
        _properties,
        false,
        ov::PropertyMutability::RW,
        [this](const ov::AnyMap&) {
            return COMPILE_LOG_LEVEL::resolve(_config);
        });
```

<br><br>

# Special cases
## SC.1 Adding a new property which requires custom functions
If the new property requires a custom callback function, only Step 4. changes.
Instead of using register_property helper function, you can choose from the following helper functions:

#### `register_property_with_support_custom_function_and_args(properties, propertyName, isPublic, mutability, isSupported, getter)`
Same as register_property_with_support_and_custom_function, but for properties whose getter also receives an ov::AnyMap of additional
arguments at get_property call time. Use this for properties such as `ov::compatibility_check` that accept
extra input arguments.
Example:
```cpp
    register_property_with_support_custom_function_and_args(
        _properties,
        ov::compatibility_check.name(),
        true,
        ov::PropertyMutability::RO,
        [this](const ov::AnyMap&) {
            return isCompatibilityCheckSupported(_backend, *_compilerOptionSupportHelper);
        },
        [this](const ov::AnyMap& arguments) {
            return validateCompatibilityDescriptor(_backend, arguments, *_compilerOptionSupportHelper);
        });
```

#### `register_property_with_custom_function` for a config-backed option
This helper function can be used whenever a custom callback function/implementation is required for this property,
provided as a lambda function. The getter receives query-time arguments as `ov::AnyMap` and must return an
`ov::Any` value. The standard callback function just returns the value from config.
Example:
```cpp
    register_property_with_custom_function<COMPILE_LOG_LEVEL>(_config, _properties, false, ov::PropertyMutability::RW,
        [this](const ov::AnyMap&) {
            return COMPILE_LOG_LEVEL::resolve(_config);
        });
```

## SC.2 Adding a new property without option which requires customization
For properties without an option, use the explicit-name overload of `register_property_with_custom_function` for an
always-supported property, or `register_property_with_support_and_custom_function` when availability depends on runtime state.

#### `register_property_with_support_and_custom_function(properties, propertyName, isPublic, mutability, isSupported, getter)`
Registers a property and gates it through isSupported.
Use this when availability depends on runtime condition (e.g. backend capability check).
Example:
```cpp
    const auto has_backend_and_valid_device = [this](const ov::AnyMap& arguments) {
        if (_backend == nullptr) {
            return false;
        }

        try {
            const auto deviceId = arguments.at(ov::device::id.name()).as<std::string>();
            return utils::getDeviceById(_backend, deviceId) != nullptr;
        } catch (...) {
            return false;
        }
    };

    register_property_with_support_and_custom_function(
        _properties,
        ov::device::full_name.name(),
        true,
        ov::PropertyMutability::RO,
        has_backend_and_valid_device,
        [this](const ov::AnyMap& arguments) {
            return utils::getFullDeviceName(_backend, arguments.at(ov::device::id.name()).as<std::string>());
        });
```

Compiler-specific properties without options, such as `COMPILER_VERSION`, use the same overload but resolve the compiler
from query-time `COMPILER_TYPE`, `DEVICE_ID`, and `PLATFORM` arguments. The plugin manager normalizes these arguments
before invoking the support predicate and getter; use `CompilerAdapterFactory` and return `false` if compiler creation fails.

#### `register_property_with_support_custom_function_and_args`
Use this when the getter needs additional query-time arguments, such as `ov::compatibility_check`:
Example:
```cpp
    register_property_with_support_custom_function_and_args(
        _properties,
        ov::compatibility_check.name(),
        true,
        ov::PropertyMutability::RO,
        [this](const ov::AnyMap&) {
            return isCompatibilityCheckSupported(_backend, *_compilerOptionSupportHelper);
        },
        [this](const ov::AnyMap& arguments) {
            return validateCompatibilityDescriptor(_backend, arguments, *_compilerOptionSupportHelper);
        });
```

## SC.3 Register options and gate availability with support checks
Register normal options in `OptionsDesc` even when their runtime or compiler support is conditional. Backend-dependent options may be registered conditionally when the option cannot exist without the required backend capability. Gate property support through the appropriate support predicate.

The plugin property manager builds the complete configuration and registers backend options through `backend->registerOptions(*options)`. Backend-dependent options can use support predicates such as `hasBackendPredicate` or a compiler support query.

For example, `WORKLOAD_TYPE` remains registered when the backend is present, while its availability is controlled by the backend capability used during option registration.
Implementation point:
src/plugins/intel_npu/src/plugin/src/plugin_property_manager.cpp > functions `register_options(...)` and `PluginPropertyManager::registerProperties()`
Example:
```cpp
    if (backend != nullptr && backend->isCommandQueueExtSupported()) {
        options.add<WORKLOAD_TYPE>();
    }
```
The logger configuration is initialized separately in `Plugin` with only `LOG_LEVEL`; the complete configuration is owned by `PluginPropertyManager`.

NPUW options exposed by `for_each_exposed_npuw_option` are registered with `register_npuw_property`. These properties are
private, read-write config-backed properties and should not be registered individually. Cached NPUW options are included
in the compiler cache-property list through `for_each_cached_npuw_option`.

## SC.4 Filter compiler configuration before serialization
Before compiler serialization in `compiler_adapter/src/model_serializer.cpp`, use the predicate-based overload of
`FilteredConfig::toStringForCompiler(...)`:
```cpp
content += config.toStringForCompiler([&isOptionSupportedByCompiler](std::string_view key) {
    return isOptionSupportedByCompiler != nullptr && isOptionSupportedByCompiler(std::string(key));
});
```
This traverses compile-time and `Both` options from the normal configuration and all internal compiler options. Only keys for which the predicate returns `true` are serialized. Runtime-only options are skipped, and unsupported options are not sent to the compiler.

<br><br>

# Removing a public property
Removing a public property can be done by removing everything added in section "How to add a new public property" step-by-step.
