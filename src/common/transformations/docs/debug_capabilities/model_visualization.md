# Model visualization

Enables visualization of the model to .svg file after each transformation pass.

## Usage

Set the `OV_ENABLE_VISUALIZE_TRACING` environment variable to `"true"`, `"on"` or `"1"` to enable visualization for all transformations.

### Filtering

You can specify filters to control which passes are visualized.
If the variable is set to a specific filter string (e.g., `"PassName"`, `"PassName1,PassName2"`),
only transformations matching that filter will be visualized. Delimiter is `,`.

Example:
```
export OV_ENABLE_VISUALIZE_TRACING=true
export OV_ENABLE_VISUALIZE_TRACING="Pass1,Pass2,Pass3"
```
