from . import OpenForecast

def test_OpenForecast():
    assert OpenForecast.apply("Jane") == "hello Jane"
