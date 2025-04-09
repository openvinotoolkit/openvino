# Transformation statistics collection and visualization

There are 3 environment variables which can be set for Transformations debugging:

1. OV_ENABLE_PROFILE_PASS - Enables profiling of transformation passes to log their execution times.


    Usage: Set this environment variable to "true" to enable visualizations.
    Alternatively, specify a file path where the execution times will be saved.

    Example:
    export OV_ENABLE_PROFILE_PASS=true
    export OV_ENABLE_PROFILE_PASS="/path/to/save/profiling/results"


2. OV_ENABLE_VISUALIZE_TRACING - Enables visualization of the model to .svg file after each transformation pass.
   

    Usage: Set this environment variable to "true", "on" or "1" to enable visualization for all Transformations.

    Filtering: You can specify filters to control which passes are visualized.
    If the variable is set to a specific filter string (e.g., "PassName", "PassName1,PassName2"),
    only transformations matching that filter will be visualized. Delimiter is ",".

    Example:
    export OV_ENABLE_VISUALIZE_TRACING=true
    export OV_ENABLE_VISUALIZE_TRACING="Pass1,Pass2,Pass3"


3. OV_ENABLE_SERIALIZE_TRACING - Enables serialization of the model to .xml/.bin after each transformation pass.
    

    Usage: Set this environment variable to "true", "on" or "1" to enable serialization for all Transformations.

    Filtering: You can specify filters to control which passes are serialized.
    If the variable is set to a specific filter string (e.g., "PassName", "PassName1,PassName2"),
    only transformations matching that filter will be serialized. Delimiter is ",".

    Example:
    export OV_ENABLE_SERIALIZE_TRACING=true
    export OV_ENABLE_SERIALIZE_TRACING="Pass1,Pass2,Pass3"

If you have suggestions for improvements or encounter any issues with statistics collection, feel free to submit your feedback or contact Ivan Tikhonov <ivan.tikhonov@intel.com>