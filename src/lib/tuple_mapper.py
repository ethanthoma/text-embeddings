from functools import wraps
from inspect import signature

def tpm(func):
    sig = signature(func)
    num_params = len(sig.parameters)

    @wraps(func)
    def wrapper(*args, **kwargs):
        if 'self' in sig.parameters:
            adjusted_num_params = num_params - 1
            # unpack tuple
            if len(args) > 1 and isinstance(args[1], tuple) and len(args[1]) == adjusted_num_params:
                return func(args[0], *args[1])
        else:
            adjusted_num_params = num_params
            if len(args) == 1 and isinstance(args[0], tuple) and len(args[0]) == adjusted_num_params:
                return func(*args[0])

        # If not enough arguments for partial application are provided, call the function as is.
        if len(args) + len(kwargs) >= adjusted_num_params:
            return func(*args, **kwargs)

        # Partial application: return a new curried function.
        @wraps(func)
        def partial(*more_args, **more_kwargs):
            new_args = args + more_args
            new_kwargs = {**kwargs, **more_kwargs}
            return wrapper(*new_args, **new_kwargs)

        return partial

    return wrapper

