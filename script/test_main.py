import pytest
import main

def test_quadroots_result():
    assert main.quad_roots(1.0, 1.0, -12.0) == ((3+0j), (-4+0j))

def test_quadroots_types():
    with pytest.raises(TypeError):
        main.quad_roots("", "green", "hi")

def test_quadroots_zerocoeff():
    with pytest.raises(ValueError):
        main.quad_roots(a=0.0)

def test_linearoots_result():
    assert main.linear_roots(2.0, -3.0) == 1.5

def test_linearroots_types():
    with pytest.raises(TypeError):
        main.linear_roots("ocean", 6.0)

def test_linearroots_zerocoeff():
    with pytest.raises(ValueError):
        main.linear_roots(a=0.0)
