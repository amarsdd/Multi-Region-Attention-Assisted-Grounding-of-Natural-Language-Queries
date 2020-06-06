import warnings
warnings.simplefilter(action='ignore', category=Warning)

def tensorflow_mute():
    """
    Make Tensorflow less verbose
    """
    try:
        import os
        from tensorflow import logging
        logging.set_verbosity(logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        def deprecated(date, instructions, warn_once=True):
            def deprecated_wrapper(func):
                return func

            return deprecated_wrapper

        from tensorflow.python.util import deprecation
        deprecation.deprecated = deprecated

    except ImportError:
        pass
tensorflow_mute()