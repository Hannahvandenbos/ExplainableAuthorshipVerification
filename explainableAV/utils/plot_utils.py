def data_path(path, manual_path):
    '''Select the correct file from two path files'''
    if manual_path is None:
        return path
    return manual_path
