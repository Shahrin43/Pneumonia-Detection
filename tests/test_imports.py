
def test_imports():
    import src.dataset as dataset
    import src.model as model
    import src.train as train
    import src.inference as inference
    assert hasattr(model, 'get_model')
