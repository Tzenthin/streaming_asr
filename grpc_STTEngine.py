# Abstract class for STT Engines

class STTEngine(object):
    def __init__(self, model_folder):
        """ Loads and sets up the model

        Args:
            model_folder: Path to folder containing model files and optional config file.

        Returns:
           A STTEngine instance.
        """
        raise NotImplementedError

    def decode_audio(self, audio):
        """ Decodes an audio segment

        Args:
            audio ([bytes]): Audio data to be transcribed.

        Returns:
            Decoded string.
        """
        raise NotImplementedError

    def get_stream(self, result_queue):
        """ Establishes stream to model.

        Args:
            result_queue:   Queue to pass completed results to.

        Returns:
            A stream object to refer back.
        """
        raise NotImplementedError

    def feed_audio_data(self, stream, audio):
        """ Adds audio data to given stream

        Args:
            stream: Object returned by setup_stream.
            audio ([bytes]): Audio data to be transcribed. 
        """
        raise NotImplementedError

    def get_partial(self, stream):
        """ Returns partial decoded result

        Args:
            stream (object returned by get_stream)

        Returns:
            Partial decoded string.
        """
        raise NotImplementedError

    def finish_stream(self, stream):
        """ Finishes decoding destroying stream.

        Args:
            stream: Object returned by setup_stream.

        """
        raise NotImplementedError

    def check_compatibility(self, config):
        """ Checks if engine is compatible with given config.

        Args:
            config: Key, value pairs of requested features.

        Returns:
            boolean, True if engine matches config.
        """
        raise NotImplementedError
