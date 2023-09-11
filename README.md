# fpsample
Python farthest point sampling library built on Rust pyo3 bindings. Compatible with numpy.

## Installation

### Install from PyPI

```shell
pip install -U fpsample
```

### Build from source

The library is built using [maturin](https://github.com/PyO3/maturin). Ensure you have maturin, rust and cargo installed.

Build the library and install using:
```shell
maturin develop --release
```
