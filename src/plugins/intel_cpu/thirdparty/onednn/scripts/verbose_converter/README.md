# Verbose log converter

Verbose log converter is a tool that allows to convert
[oneDNN verbose](https://oneapi-src.github.io/oneDNN/dev_guide_verbose.html)
output to input files for benchdnn. The tool can be extended to produce other
types of output by adding generators.

## Requirements
 - Python 3.7

## Compatibility
The script is compatible with particular oneDNN version it is distributed with.
Compatibility with other oneDNN versions is not guaranteed.
To get an appropriate version of the script:
 - Identify `DNNL_VERSION_HASH` located in `include/oneapi/dnnl/dnnl_config.h`.
 - Download the script from oneDNN repository with that particular hash.

## Usage
### Option 1: call from command line
``` sh
python3 verbose_converter.py [-h] [-i INPUT] [-p PARSER] [-a ACTION] [-s SPLIT]
                            [-v VERBOSE_LEVEL] [-o OUTPUT] [-g GENERATOR]
```
### Arguments
  - `{-h,--help}` -- display help message and exit.
  - `{-i,--input} STRING` -- input file with verbose log (default: `stdin`).
  - `{-p,--parser} oneDNN [default], ...` -- type of parser.
            Refer to ``Parsers`` below.
  - `{-a,--action} generate [default], ...` -- an action.
            Refer to ``Actions`` below.
  - `{-s,--split} BOOL` -- if `true`, generated inputs will be split between
            primitive kinds. Default is `false`.
  - `{-v,--verbose_level} N` -- verbose level. Default is `0`.
            Refer to ``Verbose`` below.
  - `{-o,--output} STRING` -- output file. Default is `stdout`. If output file
            is provided and option `-s` is set, output will be split into
            multiple files with names `driver.output`, where `driver` is a name
            of particular driver.
  - `{-g,--generator} benchdnn [default], ...` target generator.
            Refer to ``Generators`` below.

### Option 2: as Python module
``` python
import verbose_converter

output = verbose_converter.convert(verbose_level, parser, input, action,
         generator, split_output)
```
### Arguments
  - `input STRING` -- string with verbose log.
  - `parser STRING` -- type of parser.
            Refer to ``Parsers`` below.
  - `action STRING` -- an action.
            Refer to ``Actions`` below.
  - `split BOOL` -- if `true`, generated inputs will be split between
            primitive kinds. Default is `false`.
  - `verbose_level N` -- verbose level.
            Refer to ``Verbose`` below.
  - `generator STRING` -- target generator.
            Refer to ``Generators`` below.

### Return value
  - `status STRING` -- status of conversion.
            Refer to ``Statuses`` below.
  - `data` -- if `split` is `true` data is a dictionary where key is name of
            primitive. if `split` is `false` data is a list.

### Actions

| Action    | Description                             |
|:--------- |:-----------                             |
| generate  | generate input using selected generator |
| dumpIR    | dump IR generated after parsing input   |

### Generators

| Generator | Output              |
|:--------- |:------              |
| benchdnn  | benchdnn test cases |

### Parsers

| Parser | Input          |
|:------ |:-----          |
| oneDNN | oneDNN verbose |

### Statuses

| Status  | Value |
|:------  |:----- |
| SUCCESS | 0     |
| FAILED  | 1     |

### Verbose

| Level | Description                         |
|:----- |:-----------                         |
| 0     | no verbose                          |
| 1     | print verbose information to stdout |
