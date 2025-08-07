from threading import Thread
import functools


def threaded(*, is_blocking=True):
    """
    Decorator factory for non-blocking thread execution
    - daemon: Thread exits with main program (default: True)
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create and start thread
            if is_blocking:
                return func(*args, **kwargs)
            
            else:
                thread = Thread(
                    target=func,
                    args=args,
                    kwargs=kwargs,
                    daemon=True
                )
                thread.start()
            
            return thread
        return wrapper
    return decorator