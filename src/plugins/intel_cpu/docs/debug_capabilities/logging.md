## Debug log

Debug logs starting with `[ DEBUG ]` will be shown after this option is set to ON, and
each log has prefix in format `source_file_name:line_num function()` indicating the position of the log in source code.

Environment variable `OV_CPU_DEBUG_LOG` controls which debug logs to output by combining
patterns, typical examples of usages are:
   - not define it: no debug logs will be output
   - `-` : all debug logs will be output
   - `graph.cpp:798;InitEdges` :  only debug logs from "graph.cpp:798" and function "InitEdges" are output
   - `-graph.cpp:798;InitEdges` :  only debug logs from specified places are not output

Environment variable `OV_CPU_DEBUG_LOG_BRK` can be set to some keywords or a full log line seen previously, if any debug log match with the content in this variable, an `int3` instruction will be executed to trigger breakpoint trap if it's running inside a debugger.
