
# Fusing disabling
Fusing disabling is controlled by environment variables.

The variables define conditions of the node which fusing should be disabled for.

```sh
    OV_CPU_DISABLE="<filter_type>=<comma_separated_list_of_patterns> binary ...
```

Fusing handling is located in [graph_optimizer.cpp](../src/graph_optimizer.cpp)

See **enum Type** in [cpu_types.h](../src/cpu_types.h) for the list of node types

Fusing can be disabled for:

## Target node

Node which we fuse into

### By name

```sh
    OV_CPU_DISABLE="fusing_by_target_name=Convolution123,Pooling123"
```
To disable all the internal optimizations that fuse / merge any node into nodes with the names `Convolution123` OR `Pooling123`

### By type

```sh
    OV_CPU_DISABLE="fusing_by_target_type=Convolution,Pooling"
```
To disable all the internal optimizations that fuse / merge any node into nodes with the types `Convolution` OR `Pooling`

## Fused node

Node which we fuse

### By name

```sh
    OV_CPU_DISABLE="fusing_by_fused_name=Multiply123,FakeQuantize123"
```
To disable all the internal optimizations that fuse / merge nodes with the names `Multiply123` OR `Relu123` into any node

### By type
    
```sh
    OV_CPU_DISABLE="fusing_by_fused_type=Eltwise,FakeQuantize"
```
To disable all the internal optimizations that fuse / merge nodes with the types `Eltwise` OR `FakeQuantize` into any node


## Disable all the fusings

```sh
    OV_CPU_DISABLE="fusing_by_fused_name=*" binary ...
```
