import os
import datasets
import glob

# hugging face sucks

_URL = "http://www.openslr.org/12"
_DL_URL = "http://www.openslr.org/resources/12/"


_DL_URLS = {
    "clean": {
        "train_clean_100": _DL_URL + "train-clean-100.tar.gz",
        "test_clean": _DL_URL + "test-clean.tar.gz",
        "dev_clean": _DL_URL + "dev-clean.tar.gz",
    },
}


class LibrispeechASRConfig(datasets.BuilderConfig):
    """BuilderConfig for LibriSpeechASR."""

    def __init__(self, **kwargs):
        super(LibrispeechASRConfig, self).__init__(version=datasets.Version("2.1.0", ""), **kwargs)


class LibrispeechASR(datasets.GeneratorBasedBuilder):
    """Librispeech dataset."""

    DEFAULT_WRITER_BATCH_SIZE = 256
    DEFAULT_CONFIG_NAME = "clean"
    BUILDER_CONFIGS = [
        LibrispeechASRConfig(name="clean", description="'Clean' speech."),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "file": datasets.Value("string"),
                    "audio": datasets.Audio(sampling_rate=16_000),
                    "text": datasets.Value("string"),
                    "speaker_id": datasets.Value("int64"),
                    "chapter_id": datasets.Value("int64"),
                    "id": datasets.Value("string"),
                }
            ),
            supervised_keys=("file", "text"),
            homepage=_URL,
        )

    def _split_generators(self, dl_manager):
        if self.config.name == "clean":
            urls_to_download = {
                "train_clean_100": _DL_URLS["clean"]["train_clean_100"],
                "test_clean": _DL_URLS["clean"]["test_clean"],
            }
        else:
            raise ValueError(f"Configuration '{self.config.name}' not supported in this script version.")

        archive_path = dl_manager.download(urls_to_download)
        local_extracted_archive = dl_manager.extract(archive_path)

        splits = []

        splits.append(
            datasets.SplitGenerator(
                name="train_clean_100",
                gen_kwargs={
                    "path_to_extracted_archive": local_extracted_archive["train_clean_100"],
                },
            )
        )

        splits.append(
            datasets.SplitGenerator(
                name="test_clean",
                gen_kwargs={
                    "path_to_extracted_archive": local_extracted_archive["test_clean"],
                },
            )
        )
        
        return splits

    def _generate_examples(self, path_to_extracted_archive):
        
        key = 0
        
        transcription_files = glob.glob(os.path.join(path_to_extracted_archive, "**", "*.trans.txt"), recursive=True)
        
        transcriptions_by_id = {}
        for trans_file_path in transcription_files:
            if "README.TXT" in os.path.basename(trans_file_path).upper():
                continue
            
            with open(trans_file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split(" ", 1)
                        if len(parts) == 2:
                            utt_id, transcript = parts
                            transcriptions_by_id[utt_id] = transcript
                        else:
                            print(f"Warning: Skipping malformed line in {trans_file_path}: {line}")
                            
        audio_files = glob.glob(os.path.join(path_to_extracted_archive, "**", "*.flac"), recursive=True)
        
        for audio_file_path in audio_files:
            file_basename = os.path.basename(audio_file_path)
            utt_id_from_file = file_basename[: -len(".flac")]
            
            if utt_id_from_file in transcriptions_by_id:
                transcript = transcriptions_by_id[utt_id_from_file]
                
                parts = utt_id_from_file.split("-")
                if len(parts) >= 2:
                    try:
                        speaker_id = int(parts[0])
                        chapter_id = int(parts[1])
                    except ValueError:
                        print(f"Warning: Skipping utterance {utt_id_from_file} due to non-integer speaker/chapter ID.")
                        continue
                else:
                    print(f"Warning: Skipping utterance {utt_id_from_file} due to unexpected ID format.")
                    continue
                
                yield key, {
                    "file": audio_file_path,
                    "audio": audio_file_path,
                    "text": transcript,
                    "speaker_id": speaker_id,
                    "chapter_id": chapter_id,
                    "id": utt_id_from_file,
                }
                key += 1
            else:
                print(f"Warning: No transcription found for audio file {audio_file_path}. Skipping.")

