#!/usr/bin/env python3

import functools
import time

def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      # 2
        run_time = end_time - start_time    # 3
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value
    return wrapper_timer


def repeat(n=1):
	def decorator_repeat(func):
		@timer
		@functools.wraps(func)
		def wrapper_repeat(*args, **kwargs):
			for _ in range(n):
				value = func(*args, **kwargs)
			return value
		return wrapper_repeat
	return decorator_repeat


@repeat(4)
def printABCs():
	letter = 'A'
	x = [print(chr(ord(letter)+i),end="") for i in range(26)]
	print('\n')

#printABCs()

		
		