DATA_PATH='.'
PATH_OUT='.'

def test_imports(data_path='/project/ag-moonraedler/JHeyn/221111_TIRF_RPE_lanes_10min', nd2_file='221111_TIRF_RPE_lanes_10min.nd2'):

    from onedcelltrack import pipeline
    tracker = pipeline.Track(PATH_OUT, data_path=data_path, nd2_file=nd2_file, bf_channel=1, nuc_channel=0)
    return