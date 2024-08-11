gantt 
    %% Use a hack for centry as a persantage
    dateFormat YYYY
    axisFormat %y
    todayMarker off
    title       OpenVINO getting started pipeline
    Setup environment :env, 2000, 1716w
    Build openvino :crit, build, after env, 1716w
    Run tests :active, run, after build, 1716w
    # Security Policy

## Report a Vulnerability

Please report security issues or vulnerabilities to the [Intel® Security Center].

For more information on how Intel® works to resolve security issues, see
[Vulnerability Handling Guidelines].

[Intel® Security Center]:https://www.intel.com/security

[Vulnerability Handling Guidelines]:https://www.intel.com/content/www/us/en/security-center/vulnerability-handling-guidelines.html
