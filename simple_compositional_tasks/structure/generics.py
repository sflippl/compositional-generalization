import inspect

def arg_wrapper(_fun, dct, **kwargs):
    if callable(_fun):
        keys = inspect.signature(_fun).parameters
    else:
        keys = inspect.signature(_fun.__init__).parameters
    if 'kwargs' in keys:
        subset_dct = {key: value for key, value in dct.items() if key in keys and key not in kwargs}
        return _fun(**kwargs, **subset_dct)
    else:
        subset_dct = {key: value for key, value in dct.items() if key not in kwargs}
        return _fun(**kwargs, **subset_dct)

def resolve(*args):
    outp = args[0]
    for dct in args[1:]:
        for key, value in dct.items():
            if key not in outp.keys():
                outp[key] = value
    return outp

def choose_default(default, key, *dcts):
    for dct in dcts:
        if key in dct.keys():
            return dct[key]
    return default
