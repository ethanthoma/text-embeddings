from functools import wraps
from inspect import signature

def tpm(func):
    sig = signature(func)
    num_params = len(sig.parameters)

    self_num_offset = 1 if 'self' in sig.parameters else 0

    def has_len_tuple_to_map_to_args(args):
        return (
            len(args) == 1 + self_num_offset and 
            isinstance(args[self_num_offset], tuple) and 
            len(args[self_num_offset]) == num_params - self_num_offset
        )

    @wraps(func)
    def wrapper(*args, **kwargs):
        if has_len_tuple_to_map_to_args(args):
            if 'self' in sig.parameters:
                return func(args[0], *args[1])
            else:
                return func(*args[0])

        adjusted_num_params = num_params - self_num_offset

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

