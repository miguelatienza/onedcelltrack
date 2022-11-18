DATA_PATH='.'
PATH_OUT='.'

def test_imports():

    from onedcelltrack import pipeline
    tracker = pipeline.Track(PATH_OUT)
    return