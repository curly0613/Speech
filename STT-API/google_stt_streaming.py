# -*- coding: utf-8 -*-
import pyaudio

from six.moves import queue

from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types

class MicrophoneStream(object):
    """Opens a recording stream as a generator yielding the audio chunks."""
    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk
        self._frames = []

        # Create a thread-safe buffer of audio data
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            # The API currently only supports 1-channel (mono) audio
            # https://goo.gl/z757pE
            channels=1, rate=self._rate,
            input=True, frames_per_buffer=self._chunk,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't
            # overflow while the calling thread makes network requests, etc.
            stream_callback=self._fill_buffer,
        )

        self.closed = False

        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """Continuously collect data from the audio stream, into the buffer."""
        self._frames.append(in_data)
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        rcd_chk = False
        max = random.randint(1,3)
        while not self.closed:
            chunk = self._buff.get()

            if chunk is None:
                return

            data = [chunk]

            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)

                except queue.Empty:
                    break

            yield b''.join(data)

def listen_print_loop(file_name, responses, frames, rate):
    num_chars_printed = 0
    for response in responses:
        if not response.results:
            continue

        result = response.results[0]
        if not result.alternatives:
            continue

        # Display the transcription of the top alternative.
        transcript = result.alternatives[0].transcript

        overwrite_chars = ' ' * (num_chars_printed - len(transcript))

        if not result.is_final:
            #sys.stdout.write(transcript + overwrite_chars + '\r')
            #sys.stdout.flush()
            num_chars_printed = len(transcript)

        else:
            return transcript + overwrite_chars
            if re.search(r'\b(그만|끝)\b', transcript, re.I):
                print('Exiting..')
                break

            num_chars_printed = 0

def Google_Streaming_STT(file_name, contexts=[' '], record_time=10) :
    RATE = 16000
    CHUNK = int(RATE / 10)  # 100ms

    client = speech.SpeechClient()
    config = types.RecognitionConfig(
        encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code='ko-KR', speech_contexts=[speech.types.SpeechContext(phrases=contexts)])
    streaming_config = types.StreamingRecognitionConfig(config=config, interim_results=True)

    result = ''
    print("========== Start Streaming API")
    with MicrophoneStream(RATE, CHUNK) as stream:
        audio_generator = stream.generator()
        requests = (types.StreamingRecognizeRequest(audio_content=content) for content in audio_generator)

        responses = client.streaming_recognize(streaming_config, requests)
        result = listen_print_loop(file_name, responses, stream._frames, stream._rate)
    print("========== Finish Streaming API")
    return result


if __name__ == '__main__':
    Google_Streaming_STT("google_record.wav", contexts=[' '], record_time=10)
