# Debug capabilities
Use the following cmake option to enable debug capabilities:

`-DENABLE_DEBUG_CAPS=ON`

* [Verbose mode](verbose.md)
* [Blob dumping](blob_dumping.md)
* [Graph serialization](graph_serialization.md)

## Debug log

Debug logs starting with `[ DEBUG ]` will be shown after this option is set to ON, and
each log will be start with `function_name:line_num` indicating the position of the log
in source code.

Environment variable `OV_CPU_DEBUG_LOG` controls which debug logs to output by combining
patterns of `function_name` or `function_name:line_num`, typical examples of usages are:
   - not define it: no debug logs will be output
   - `-` : all debug logs will be output
   - `foo;bar:line2` :  only debug logs at "foo:*" and "bar:line2" are output
   - `-foo;bar:line2` :  only debug logs at "foo:*" and "bar:line2" are not output

## Performance summary
set `OV_CPU_SUMMARY_PERF` environment variable to display performance summary at the time when model is being destructed.

Internal performance counter will be enabled automatically. 
