from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess


class SenseVoiceInferencer:
    def __init__(self, model_path):
        model_dir = f"{model_path}/SenseVoiceSmall"

        # ModelScope VAD may be unavailable in some environments; try VAD first,
        # fall back to no VAD (whole-utterance ASR still works for WER).
        try:
            model = AutoModel(
                model=model_dir,
                vad_model="fsmn-vad",
                vad_kwargs={"max_single_segment_time": 30000},
                device="cuda:0",
                hub="hf",
                disable_update=True,
            )
            self._use_vad = True
        except Exception as e:
            print(f"[warn] SenseVoice VAD init failed; disabling VAD: {e}")
            model = AutoModel(
                model=model_dir,
                vad_model=None,
                device="cuda:0",
                hub="hf",
                disable_update=True,
            )
            self._use_vad = False
        self.model = model

    def infer(self, audio_path):
        gen_kwargs = dict(
            input=audio_path,
            cache={},
            language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
            use_itn=True,
            batch_size_s=60,
        )
        if self._use_vad:
            gen_kwargs.update(dict(merge_vad=True, merge_length_s=15))
        res = self.model.generate(**gen_kwargs)
        text = rich_transcription_postprocess(res[0]["text"])
        return text

