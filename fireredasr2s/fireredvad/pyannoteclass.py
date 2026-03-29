import torch
import torchaudio
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.audio.telemetry import set_telemetry_metrics

# disable metrics
set_telemetry_metrics(False, save_choice_as_default=True)

# from pydub import AudioSegment
import io

class PyannoteModel:

    def __init__(
        self, 
        model_path: str,
        use_gpu: int,
    ):
        self.pipeline = Pipeline.from_pretrained(
            model_path,
        )
        if use_gpu:
            self.pipeline.to(torch.device("cuda"))
    
    def __call__(
        self, 
        input_file: str,
        num_speakers: int | None = None
    ):
        waveform, sample_rate = torchaudio.load(input_file)

        audio_in_memory = {"waveform": waveform, "sample_rate": sample_rate}

        with torch.inference_mode():
            output = self.pipeline(
                audio_in_memory,
                num_speakers=num_speakers,
            )

        return output.serialize()