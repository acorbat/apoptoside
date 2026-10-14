import importlib

import pytest

MODULES = [
    "anisotropy_functions",
    "apoptoside",
    "experimental_tests",
    "filter_data",
    "model",
    "sensors",
    "transformation",
    "viewer",
]


@pytest.mark.parametrize("name", MODULES)
def test_import(name):
    importlib.import_module(f"apoptoside.{name}")
