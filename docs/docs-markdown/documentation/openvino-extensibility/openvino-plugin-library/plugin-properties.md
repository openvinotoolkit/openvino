---
sidebar_position: 5
sidebar_label: "Plugin Properties"
description: "Use the ov::Property class to define access rights and specific properties of an OpenVINO plugin."
---

# Plugin Properties

Plugin can provide own device-specific properties.

## Property Class

OpenVINO API provides the interface ov::Property which allows to define the property and access rights. Based on that, a declaration of plugin specific properties can look as follows:

```cpp

/**
 * @brief Allows to disable all transformations for execution inside the TEMPLATE plugin.
 */
static constexpr Property<bool, PropertyMutability::RW> disable_transformations{"DISABLE_TRANSFORMATIONS"};
```
