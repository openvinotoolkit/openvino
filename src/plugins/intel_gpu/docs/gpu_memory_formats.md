# GPU Memory Formats

The memory format descriptor in GPU plugin usually uses the following letters:
 - `b` - batch
 - `f` - features/channels
 - `w`, `z`, `y`, `x` - spatial dimensions
 - `i` - input channels (for weights layout only)
 - `o` - output channels (for weights layout only)
 - `g` - groups (for weights layout only)

The combination of the characters above defines the tensor format, that is, the actual layout of tensor values in a memory buffer. For example:
`bfyx` format means that the tensor has 4 dimensions in planar layout and `x` coordinate changes faster than `y`, `y` - faster than `f`, and so on.
It means that for a tensor with size `[b: 2; f: 2; y: 2; x: 2]`, there is a linear memory buffer with `size=16`, where:
```
i = 0  => [b=0; f=0; y=0; x=0];
i = 1  => [b=0; f=0; y=0; x=1];

i = 2  => [b=0; f=0; y=1; x=0];
i = 3  => [b=0; f=0; y=1; x=1];

i = 4  => [b=0; f=1; y=0; x=0];
i = 5  => [b=0; f=1; y=0; x=1];

i = 6  => [b=0; f=1; y=1; x=0];
i = 7  => [b=0; f=1; y=1; x=1];

i = 8  => [b=1; f=0; y=0; x=0];
i = 9  => [b=1; f=0; y=0; x=1];

i = 10 => [b=1; f=0; y=1; x=0];
i = 11 => [b=1; f=0; y=1; x=1];

i = 12 => [b=1; f=1; y=0; x=0];
i = 13 => [b=1; f=1; y=0; x=1];

i = 14 => [b=1; f=1; y=1; x=0];
i = 15 => [b=1; f=1; y=1; x=1];
```

Usually, planar memory formats are not very efficient for DNN operations, so GPU plugin has plenty of *blocked* formats. Blocking means that you take some tensor dimension
and put blocks of adjacent elements closer in memory (in the format with a single blocking, they are stored linearly in the memory). Consider the most widely used
blocked format in GPU plugin: `b_fs_yx_fsv16`. First of all, let's understand what these additional letters mean. There are `b`, `f`, `y`, `x` dimensions here, so
this is a 4D tensor.
`fs=CeilDiv(f, block_size)`; `fs` means `feature slice` - the blocked dimension.
The block size is specified in the format name: `fsv16` - `block_size = 16`, a blocked dimension is `f`; `fsv` means `feature slice vector`
Just like with any other layout, the coordinate of the rightmost dimension (`fsv`) is changed first, then coordinate to the left (`x`), and so on.

> **Note**: If the original `f` dimension is not divisible by block size (`16` in this case), then it is aligned up to the first divisible value. These pad values
are filled with zeroes.

When you reorder the tensor above into `b_fs_yx_fsv16` format, changes are as follows:
1. Actual buffer size becomes `[b: 2; f: 16; y: 2; x: 2]`, and total size equals 128.
2. The order of elements in memory changes:
```
// first batch
i = 0   => [b=0; f=0;  y=0; x=0] == [b=0; fs=0; y=0; x=0; fsv=0];
i = 1   => [b=0; f=1;  y=0; x=0] == [b=0; fs=0; y=0; x=0; fsv=1];
i = 2   => [b=0; f=2;  y=0; x=0] == [b=0; fs=0; y=0; x=0; fsv=2];
...
i = 15  => [b=0; f=15; y=0; x=0] == [b=0; fs=0; y=0; x=0; fsv=15];

i = 16  => [b=0; f=0;  y=0; x=1] == [b=0; fs=0; y=0; x=1; fsv=0];
i = 17  => [b=0; f=1;  y=0; x=1] == [b=0; fs=0; y=0; x=1; fsv=1];
i = 18  => [b=0; f=2;  y=0; x=1] == [b=0; fs=0; y=0; x=1; fsv=2];
...
i = 31  => [b=0; f=15; y=0; x=1] == [b=0; fs=0; y=0; x=1; fsv=15];

i = 32  => [b=0; f=0;  y=1; x=0] == [b=0; fs=0; y=1; x=0; fsv=0];
i = 33  => [b=0; f=1;  y=1; x=0] == [b=0; fs=0; y=1; x=0; fsv=1];
i = 34  => [b=0; f=2;  y=1; x=0] == [b=0; fs=0; y=1; x=0; fsv=2];
...
i = 47  => [b=0; f=15; y=1; x=0] == [b=0; fs=0; y=1; x=0; fsv=15];

i = 48  => [b=0; f=0;  y=1; x=1] == [b=0; fs=0; y=1; x=1; fsv=0];
i = 49  => [b=0; f=1;  y=1; x=1] == [b=0; fs=0; y=1; x=1; fsv=1];
i = 50  => [b=0; f=2;  y=1; x=1] == [b=0; fs=0; y=1; x=1; fsv=2];
...
i = 63  => [b=0; f=15; y=1; x=1] == [b=0; fs=0; y=1; x=1; fsv=15];

// second batch
i = 64  => [b=1; f=0;  y=0; x=0] == [b=1; fs=0; y=0; x=0; fsv=0];
i = 65  => [b=1; f=1;  y=0; x=0] == [b=1; fs=0; y=0; x=0; fsv=1];
i = 66  => [b=1; f=2;  y=0; x=0] == [b=1; fs=0; y=0; x=0; fsv=2];
...
i = 79  => [b=1; f=15; y=0; x=0] == [b=1; fs=0; y=0; x=0; fsv=15];

i = 80  => [b=1; f=0;  y=0; x=1] == [b=1; fs=0; y=0; x=1; fsv=0];
i = 81  => [b=1; f=1;  y=0; x=1] == [b=1; fs=0; y=0; x=1; fsv=1];
i = 82  => [b=1; f=2;  y=0; x=1] == [b=1; fs=0; y=0; x=1; fsv=2];
...
i = 95  => [b=1; f=15; y=0; x=1] == [b=1; fs=0; y=0; x=1; fsv=15];

i = 96  => [b=1; f=0;  y=1; x=0] == [b=1; fs=0; y=1; x=0; fsv=0];
i = 97  => [b=1; f=1;  y=1; x=0] == [b=1; fs=0; y=1; x=0; fsv=1];
i = 98  => [b=1; f=2;  y=1; x=0] == [b=1; fs=0; y=1; x=0; fsv=2];
...
i = 111 => [b=1; f=15; y=1; x=0] == [b=1; fs=0; y=1; x=0; fsv=15];

i = 112 => [b=1; f=0;  y=1; x=1] == [b=1; fs=0; y=1; x=1; fsv=0];
i = 113 => [b=1; f=1;  y=1; x=1] == [b=1; fs=0; y=1; x=1; fsv=1];
i = 114 => [b=1; f=2;  y=1; x=1] == [b=1; fs=0; y=1; x=1; fsv=2];
...
i = 127 => [b=1; f=15; y=1; x=1] == [b=1; fs=0; y=1; x=1; fsv=15];
```

All formats used by GPU plugin are specified in `src/plugins/intel_gpu/include/intel_gpu/runtime/format.hpp` file. Most of the formats there follow the notation above.

## See also

 * [OpenVINO™ README](../../../../README.md)
 * [OpenVINO Core Components](../../../README.md)
 * [OpenVINO Plugins](../../README.md)
 * [OpenVINO GPU Plugin](../README.md)
 * [Developer documentation](../../../../docs/dev/index.md)