import os
import sys

from utils import make_no_metadata_graph, to_amr_with_pointer
sys.path.append(os.path.join(os.path.dirname(__file__), "AMRBART-id/fine-tune"))

from common.options import DataTrainingArguments, ModelArguments, Seq2SeqTrainingArguments
from data_interface.dataset import AMR2TextDataSet, DataCollatorForAMR2Text
import datasets
from datasets import load_from_disk
import json
import logging
from model_interface.tokenization_bart import AMRBartTokenizer
import penman
from seq2seq_trainer import Seq2SeqTrainer
import torch
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    MBartTokenizer,
    MBartTokenizerFast,
    MBartForConditionalGeneration as BartForConditionalGeneration,
    set_seed,
    T5TokenizerFast,
)
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

def mkdir_if_not_exists(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def prepare_tokenizer_and_model(training_args, model_args, data_args):
    # Set seed before initializing model.
    set_seed(training_args.seed)

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None
    )

    tokenizer = AMRBartTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    model = BartForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    model.resize_token_embeddings(len(tokenizer))

    # config dec_start_token, max_pos_embeddings
    if model.config.decoder_start_token_id is None and isinstance(
        tokenizer, (MBartTokenizer, MBartTokenizerFast)
    ):
        if isinstance(tokenizer, MBartTokenizer):
            model.config.decoder_start_token_id = tokenizer.lang_code_to_id[data_args.lang]
        else:
            model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(data_args.lang)

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    if (
        hasattr(model.config, "max_position_embeddings")
        and model.config.max_position_embeddings < data_args.max_source_length
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                "Increasing the model's number of position embedding vectors from"
                f" {model.config.max_position_embeddings} to {data_args.max_source_length}."
            )
            model.resize_position_embeddings(data_args.max_source_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has"
                f" {model.config.max_position_embeddings} position encodings. Consider either reducing"
                f" `--max_source_length` to {model.config.max_position_embeddings} or to automatically resize the"
                " model's position encodings by passing `--resize_position_embeddings`."
            )

    if training_args.label_smoothing_factor > 0 and not hasattr(
        model, "prepare_decoder_input_ids_from_labels"
    ):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )
        
    return tokenizer, model

def prepare_data_args(data_path):
    data_cache_parent = f"{data_path}/.cache"
    mkdir_if_not_exists(data_cache_parent)
    data_cache = f"{data_cache_parent}/dump-amrparsing"
    # TODO for future: Export HF_DATASETS_CACHE with data_cache here.
    mkdir_if_not_exists(data_cache)

    data_args = DataTrainingArguments(
        data_dir=data_path,
        unified_input=True,
        test_file=f"{data_path}/inference.jsonl",
        data_cache_dir=data_cache,
        overwrite_cache=True,
        max_source_length=1024,
        max_target_length=384
    )
    
    return data_args

def setup_logging(log_level):
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

class AMRToTextBase:
    def __call__(self, graphs: list[penman.Graph]) -> list[str]:
        """
        Transform all AMR graphs into sentences.

        Args:
        - `graphs`: List of AMR graph.
        """

        raise NotImplementedError("No implementation for base class.")

DEFAULT_ROOT_DIR = os.path.join(os.path.dirname(__file__), "AMRBART-id")
class AMRToText(AMRToTextBase):
    """
    Class for transforming AMR to text, a.k.a. AMR generation. This is a simplified version of
    `AMRBART-id/fine-tune/main.py` from Nafkhan.
    """

    def __init__(
            self,
            model_name: str,
            root_dir: str = DEFAULT_ROOT_DIR,
            dataset: str = "wrete",
            logging_at_training_process_level: bool = False,
            no_tqdm: bool = True,
    ):
        """
        Initialize `AMRToText` class.

        Args:
        - `model_name`: Model name which the model can be referred to `<root_dir>/models/<model_name>`

        - `root_dir`: Root of directory, used for store data, model, or cache.

        - `dataset`: Name of folder which contains `inference.json` that will be used for test dataset.

        - `logging_at_training_process_level`: If it's `True`, there's a lot of log.

        - `no_tqdm`: If it's `True`, it will disable (or reduce) TQDM progress bar.
        """

        output_dir_parent = f"{root_dir}/outputs"
        mkdir_if_not_exists(output_dir_parent)

        output_dir = f"{output_dir_parent}/infer-{model_name}"
        mkdir_if_not_exists(output_dir)

        batch_size=1
        self.training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            do_predict=True,
            logging_dir=f"{output_dir}/logs",
            seed=42,
            dataloader_num_workers=1,
            report_to="tensorboard",
            ddp_find_unused_parameters=False,
            dataloader_pin_memory=True,
            include_inputs_for_metrics=True,

            smart_init=False,
            predict_with_generate=True,
            task="amr2text",
            generation_max_length=384,
            generation_num_beams=5,

            disable_tqdm=no_tqdm,

            # Issue: Do we really need this?
            per_device_eval_batch_size=batch_size,
            eval_dataloader_num_workers=1,
        )

        if logging_at_training_process_level:
            setup_logging(self.training_args.get_process_log_level())

        self.model_args = ModelArguments(
            model_name_or_path=f"{root_dir}/models/{model_name}",
            cache_dir=f"{root_dir}/.cache",
            use_fast_tokenizer=False
        )

        data_path_parent = f"{root_dir}/ds"
        mkdir_if_not_exists(data_path_parent)
        data_path = f"{data_path_parent}/{dataset}"
        mkdir_if_not_exists(data_path)
        self.data_args = prepare_data_args(data_path)

        self.tokenizer, self.model = prepare_tokenizer_and_model(self.training_args, self.model_args, self.data_args)

        self.data_collator = DataCollatorForAMR2Text(
            self.tokenizer,
            label_pad_token_id=self.tokenizer.pad_token_id,
            pad_to_multiple_of=8 if self.training_args.fp16 else None,
        )

        self.model_name = model_name

        if no_tqdm:
            datasets.disable_progress_bars()
        else:
            datasets.enable_progress_bars()

    def __call__(self, graphs: list[penman.Graph]) -> list[str]:
        """
        Transform all AMR graphs into sentences.

        Args:
        - `graphs`: List of AMR graph.
        """
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=self.training_args,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator
        )

        predict_dataset = self._prepare_predict_dataset(graphs)
        max_length = self.training_args.generation_max_length

        num_beams = (
            self.data_args.num_beams
            if self.data_args.num_beams is not None
            else self.training_args.generation_num_beams
        )

        predict_results = trainer.predict(
            predict_dataset, metric_key_prefix="predict", max_length=max_length, num_beams=num_beams
        )
        return self._make_sentences(predict_results.predictions)

    def _make_sentences(self, predictions) -> list[str]:
        decoded_preds: list[str] = []
        for i in range(len(predictions)):
            try:
                p = predictions[i:i+1]
                single_decoded_preds = self.tokenizer.batch_decode(p, skip_special_tokens=True)
                decoded_preds.append(single_decoded_preds[0].strip())
            except Exception as e:
                print(f"Error when processing this prediction (index: {i}):\n {p[i]}")
                print("Error:", e)
                decoded_preds.append("")
        
        return decoded_preds

    @staticmethod
    def from_huggingface(repo_id: str, model_name: str, root_dir: str = DEFAULT_ROOT_DIR, hf_kwargs: dict = {}, **kwargs):
        """
        [DEPRECATED]

        Load model from Huggingface. Basically, it uses `huggingface_hub.snapshot_download`. Make sure
        `huggingface_hub` has been installed since it's an optional library.

        Args:
        - `repo_id`: Huggingface repository ID. Repository should contains a folder that contains text-to-AMR model.

        - `model_name`: This model name should related to a folder name that contains text-to-AMR model.

        - `root_dir`: Root of directory, used for store data, model, or cache.

        - `hf_kwargs`: Any other arguments that want to be passed into `huggingface_hub.snapshot_download`.

        - `kwargs`: Any other arguments that want to be passed for `AMRToText` initialization.
        """
        print("Warning: This is deprecated. Load manually instead by using this script:\nfrom huggingface_hub import snapshot_download\n\nsnapshot_download(repo_id=repo_id, **hf_kwargs)\nmodel = AMRToText(model_name)")

        from huggingface_hub import snapshot_download
        if "local_dir" not in hf_kwargs:
            hf_kwargs["local_dir"] = f"{root_dir}/models"
        
        snapshot_download(repo_id=repo_id, **hf_kwargs)

        return AMRToText(model_name, root_dir, **kwargs)
    
    def _prepare_predict_dataset(self, graphs: list[penman.Graph]):
        data_args = self.data_args
        with open(data_args.test_file, encoding="utf-8", mode="w") as fp:
            for g in graphs:
                no_metadata_g = make_no_metadata_graph(g)
                pointer_g = to_amr_with_pointer(penman.encode(no_metadata_g, indent=None))

                # sent is unused, but it's not empty to avoid error
                json_str = json.dumps({"sent": "-", "amr": pointer_g, "lang": "id"})
                print(json_str, file=fp)

        raw_datasets = AMR2TextDataSet(self.tokenizer, data_args, self.model_args)
        column_names = raw_datasets.datasets["test"].column_names

        if "test" not in raw_datasets.datasets:
            raise ValueError("--do_predict requires a test dataset")

        predict_dataset = raw_datasets.datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        if data_args.overwrite_cache or not os.path.exists(data_args.data_cache_dir + "/test"):
            with self.training_args.main_process_first(desc="prediction dataset map pre-processing"):
                predict_dataset = predict_dataset.map(
                    raw_datasets.tokenize_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on prediction dataset",
                )
                predict_dataset.save_to_disk(data_args.data_cache_dir + "/test")
        else:
            predict_dataset = load_from_disk(data_args.data_cache_dir + "/test", keep_in_memory=True)

        return predict_dataset


T5_PREFIX = "translate graph to indonesian: "

class AMRToTextWithTaufiqMethod(AMRToTextBase):
    """
    Class for transforming AMR to text, a.k.a. AMR generation, with Taufiq method
    ([code](https://github.com/taufiqhusada/amr-to-text-indonesia)).
    """

    def __init__(
            self,
            model_path: str,
            lowercase: bool = True,
            num_beams: int = 5,
            max_length: int = 384,
    ):
        """
        Initialize `AMRToTextWithTaufiqMethod` class.

        Args:
        - `model_path`: Model path. Make sure it contains model and tokenizer folders.`

        - `lowercase`: Is the model can only accept lowercase inputs?

        - `num_beams`: Number of beams used for generation.

        - `max_length`: Maximum length of prediction tokens.
        """

        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            print("Running on the GPU")
        else:
            device = torch.device("cpu")
            print("Running on the CPU")

        self.device = device

        self.tokenizer = T5TokenizerFast.from_pretrained(os.path.join(model_path, 'tokenizer'))

        model = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(model_path, 'model'))
        for param in model.parameters():
            param.data = param.data.contiguous()

        #moving the model to device(GPU/CPU)
        model.to(device)
        model.eval()

        self.model = model
        self.lowercase = lowercase
        self.num_beams = num_beams
        self.max_length = max_length

    def __call__(self, graphs: list[penman.Graph]) -> list[str]:
        """
        Transform all AMR graphs into sentences.

        Args:
        - `graphs`: List of AMR graph.
        """
        sentences: list[str] = []

        for g in tqdm(graphs):
            no_metadata_g = make_no_metadata_graph(g)
            text = to_amr_with_pointer(
                penman.encode(no_metadata_g, indent=None)
            )

            if self.lowercase:
                text = text.lower()
            
            input_ids = self.tokenizer.encode(
                f"{T5_PREFIX}{text}",
                return_tensors="pt",
                add_special_tokens=False
            )  # Batch size 1

            input_ids = input_ids.to(self.device)
            outputs = self.model.generate(
                input_ids,
                num_beams=self.num_beams,
                max_length=self.max_length
            )

            gen_text: str = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            sentences.append(gen_text)
        
        return sentences
