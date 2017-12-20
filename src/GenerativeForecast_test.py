from . import GenerativeForecast

def test_GenerativeForecast():
    assert GenerativeForecast.apply("Jane") == "hello Jane"
