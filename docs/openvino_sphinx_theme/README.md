# openvino_sphinx_theme

## Installation and usage

1. Install the `openvino_sphinx_theme` using `python`:
```
cd openvino/docs/openvino_sphinx_theme
python -m pip install --user .
```

2. Update the `html_theme` variable in your `conf.py`:

```
html_theme = 'openvino_sphinx_theme'
```

## Configuration

### Theme Logo

To add a logo at the left of your navigation bar, use `html_logo` variable to set the path to the logo file.

```
html_logo = <path to the logo file>
```

### Version and language selectors

To enable a version and language selectors, add the following configuration to your `conf.py` in `html_context`:

```
html_context = {
    'current_version': 'latest',
    'current_language': 'en',
    'languages': (('English', '/en/latest/'), ('Chinese', '/cn/latest/')),
    'versions': (('latest', '/en/latest/'), ('2022.1', '/en/2022.1'))
}
```

You can add selectors only for versions or languages.
If you want to add version selector you must define both `current_version` and `versions` properties.
If you want to add version selector you must define both `current_language` and `languages` properties.


### Maintainers

* OpenVINO Documentation team
