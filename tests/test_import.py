def test_import():
    try:
        import learning_gradient_flow
    except ImportError:
        assert False, "Failed to import $dir_name"
    assert True
