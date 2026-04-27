# Model serialization

Enables serialization of the model to .xml/.bin after each transformation pass.

## Usage

Set the `OV_ENABLE_SERIALIZE_TRACING` environment variable to `"true"`, `"on"` or `"1"` to enable serialization for all transformations.

### Filtering

You can specify filters to control which passes are serialized.
If the variable is set to a specific filter string (e.g., `"PassName"`, `"PassName1,PassName2"`),
only transformations matching that filter will be serialized. Delimiter is `,`.

Example:
```
export OV_ENABLE_SERIALIZE_TRACING=true
export OV_ENABLE_SERIALIZE_TRACING="Pass1,Pass2,Pass3"
```