from threading import Thread
import functools


def threaded(*, daemon=True):
    """
    Decorator factory for non-blocking thread execution
    - daemon: Thread exits with main program (default: True)
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create and start thread
            thread = Thread(
                target=func,
                args=args,
                kwargs=kwargs,
                daemon=daemon
            )
            thread.start()
            
            return thread
        return wrapper
    return decorator