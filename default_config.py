import os
from dataclasses import dataclass

@dataclass
class FireRedASR2Config:
    # fireredasr2s.asr_system

    """ Module Switches """
    enable_vad: int = 1
    enable_lid: int = 0
    enable_punc: int = 1
    enable_spk: int = 1

    """ ASR Options """
    # asr_type: str = "aed" # ["aed", "llm"]
    asr_type: str = os.environ.get("ASR_TYPE", "aed") # ["aed", "llm"]

    asr_model_dir: str = "/workspace/models/FireRedASR2S/FireRedASR2-AED" # asr_type == "aed"
    # asr_model_dir: str = os.environ.get("ASR_MODEL", "/workspace/models/FireRedASR2S/FireRedASR2-AED")
    asr_use_gpu: int = 1
    asr_use_half: int = 0
    asr_batch_size: int = 1

    # FireRedASR-AED
    beam_size: int = 3
    decode_max_len: int = 0
    nbest: int = 1
    softmax_smoothing: float = 1.25
    aed_length_penalty: float = 0.6
    eos_penalty: float = 1.0
    return_timestamp: int = 1 if os.environ.get("ASR_TYPE", "aed") == "aed" else 0

    # FireRedASR-AED External LM
    elm_dir: str = ""
    elm_weight: float = 0.0

    """ VAD Options """
    vad_model_dir: str = "/workspace/models/FireRedASR2S/FireRedVAD/VAD"
    vad_use_gpu: int = 1

    vad_chunk_max_frame: int = 30000
    smooth_window_size: int = 5
    speech_threshold: float = 0.2
    min_speech_frame: int = 20
    max_speech_frame: int = 1000
    
    min_silence_frame: int = 10
    merge_silence_frame: int = 50
    extend_speech_frame: int = 10

    """ LID Options """
    lid_model_dir: str = "/workspace/models/FireRedASR2S/FireRedLID"
    lid_use_gpu: int = 1

    """ Punc Options """
    punc_model_dir: str = "/workspace/models/FireRedASR2S/FireRedPunc"
    punc_use_gpu: int = 1
    punc_batch_size: int = 1
    punc_with_timestamp: int = 1
    punc_sentence_max_length: int = -1

    """ Spk Options """
    spk_model_dir: str = "/workspace/models/FireRedASR2S/pyannote/speaker-diarization-community-1"
    spk_mode: str = os.environ.get("SPK_MODE", "vad")
    






