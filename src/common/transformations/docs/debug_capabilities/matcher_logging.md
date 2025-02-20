# Logging of the pattern matching

The logging functionality allows to observe/debug the pattern matching process.

## Usage
In order to utilzie the logging, first, you need to set the CMake flag ```-DENABLE_OPENVINO_DEBUG=ON```

_NOTE: the logging would also work if your build is configured as Release_

In order to start logging, set the environmental variable ```OV_MATCHER_LOGGING``` to ``true/ON`` before running your executable or script as following:
```OV_MATCHER_LOGGING=true ./your_amazing_program```            

If you want to log only specific matchers, use the ```OV_MATCHERS_TO_LOG``` environmental variable and provide their names separated as commas:
```OV_MATCHER_LOGGING=true OV_MATCHERS_TO_LOG=EliminateSplitConcat,MarkDequantization ./your_amazing_program```

You can also set the environmental variable ```OV_VERBOSE_LOGGING``` to ```true```, to turn on more verbose logging that would print more information about the nodes taking part in the matching process:
```OV_MATCHER_LOGGING=true OV_VERBOSE_LOGGING=true ./your_amazing_program```

## Useful feature:
If you redirect the logging output into a .json file (extension is important for the correct processing of the file in Visual Studio Code). This allows collapsing blocks in Visual Studio Code as you would do with a normal .json file.

If you have any suggestions for improvement or you observe a bug in logging, feel free to submit changes or contact Andrii Staikov <andrii.staikov@intel.com>