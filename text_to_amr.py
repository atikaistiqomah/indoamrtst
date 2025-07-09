import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "AMRBART-id/fine-tune"))

from common.options import DataTrainingArguments, ModelArguments, Seq2SeqTrainingArguments
from data_interface.dataset import AMRParsingDataSet, DataCollatorForAMRParsing
import datasets
from datasets import load_from_disk
import json
import logging
from model_interface.tokenization_bart import AMRBartTokenizer
import penman
from seq2seq_trainer import Seq2SeqTrainer
import transformers
from transformers import (
    AutoConfig,
    MBartTokenizer,
    MBartTokenizerFast,
    MBartForConditionalGeneration as BartForConditionalGeneration,
    set_seed
)

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
        max_source_length=400,
        max_target_length=1024
    )
    
    return data_args

def setup_logging(log_level):
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

DEFAULT_ROOT_DIR = os.path.join(os.path.dirname(__file__), "AMRBART-id")
class TextToAMR:
    """
    Class for transforming text to AMR, a.k.a. AMR parsing. This is a simplified version of
    `AMRBART-id/fine-tune/main.py` from Nafkhan.
    """

    def __init__(
            self,
            model_name: str,
            root_dir: str = DEFAULT_ROOT_DIR,
            dataset: str = "wrete",
            logging_at_training_process_level: bool = False,
            annotator: str | None = None,
            use_prefix: bool = False
    ):
        """
        Initialize `TextToAMR` class.

        Args:
        - `model_name`: Model name which the model can be referred to `<root_dir>/models/<model_name>`

        - `root_dir`: Root of directory, used for store data, model, or cache.

        - `dataset`: Name of folder which contains `inference.json` that will be used for test dataset.

        - `logging_at_training_process_level`: If it's `True`, there's a lot of log.

        - `annotator`: If it's `None`, the value will be same as `model_name`.

        - `use_prefix`: If it's True, every sentences will be prefixed with `id_ID`. This is for compatibility.

        - `scale_embedding`: This is for compatibility.
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
            task="text2amr",
            generation_max_length=1024,
            generation_num_beams=5,

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

        self.tokenizer, self.model = prepare_tokenizer_and_model(
            self.training_args,
            self.model_args,
            self.data_args
        )

        self.data_collator = DataCollatorForAMRParsing(
            self.tokenizer,
            label_pad_token_id=self.tokenizer.pad_token_id,
            pad_to_multiple_of=8 if self.training_args.fp16 else None,
        )

        self.max_src_length = min(self.data_args.max_source_length, self.tokenizer.model_max_length)
        self.max_gen_length = (
            self.training_args.generation_max_length
            if self.training_args.generation_max_length is not None
            else self.data_args.val_max_target_length
        )

        if annotator is None:
            self.annotator = model_name
        else:
            self.annotator = annotator

        self.use_prefix = use_prefix

    def __call__(self, sentences: list[str], method: int = 1) -> list[penman.Graph]:
        """
        Transform all sentences into AMR graphs.

        Args:
        - `sentences`: List of sentence.
        """
        if method == 1:
            return self._call_method_1(sentences)
        elif method == 2:
            return self._call_method_2(sentences)
        else:
            raise ValueError(f"No method {method}")
        
        # TODO: Handle removal of prefix in graphs

    def _call_method_1(self, sentences):
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=self.training_args,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator
        )

        predict_dataset = self._prepare_predict_dataset(sentences)
        max_length = self.max_gen_length
        num_beams = self.training_args.generation_num_beams

        predict_results = trainer.predict(
            predict_dataset, metric_key_prefix="predict", max_length=max_length, num_beams=num_beams
        )

        graphs = self._make_graphs(predict_results.predictions)
        assert len(graphs) == len(sentences), f"Inconsistent lengths for graphs ({len(graphs)}) vs sentences ({len(sentences)})"

        idx = 0
        for gp, snt in zip(graphs, sentences):
            metadata = {}
            metadata["id"] = str(idx)
            metadata["annotator"] = self.annotator
            metadata["snt"] = snt
            gp.metadata = metadata
            idx += 1

        return graphs
    
    def _call_method_2(self, sentences):
        raw_txt_ids = self.tokenizer(
            sentences,
            max_length=self.max_src_length,
            padding=False,
            truncation=True
        )["input_ids"]
        txt_ids = [itm[:self.max_src_length-3] + [
            self.tokenizer.amr_bos_token_id,
            self.tokenizer.mask_token_id,
            self.tokenizer.amr_eos_token_id
        ] for itm in raw_txt_ids]
        txt_ids = self.tokenizer.pad(
            {"input_ids": txt_ids},
            padding=True,
            pad_to_multiple_of=None,
            return_tensors="pt",
            device=self.model.device
        )
        preds = self.model.generate(
            txt_ids["input_ids"],
            max_length=self.max_gen_length,
            num_beams=self.training_args.generation_num_beams,
            use_cache=True,
            decoder_start_token_id=self.tokenizer.amr_bos_token_id,
            eos_token_id=self.tokenizer.amr_eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            no_repeat_ngram_size=0,
            min_length=0,
            length_penalty=self.training_args.eval_lenpen
        )
        graphs = self._make_graphs(preds)
        assert len(graphs) == len(sentences), f"Inconsistent lengths for graphs ({len(graphs)}) vs sentences ({len(sentences)})"

        idx = 0
        for gp, snt in zip(graphs, sentences):
            metadata = {}
            metadata["id"] = str(idx)
            metadata["annotator"] = self.annotator
            metadata["snt"] = snt
            gp.metadata = metadata
            idx += 1

        return graphs

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

        - `kwargs`: Any other arguments that want to be passed for `TextToAMR` initialization.
        """
        print("Warning: This is deprecated. Load manually instead by using this script:\nfrom huggingface_hub import snapshot_download\n\nsnapshot_download(repo_id=repo_id, **hf_kwargs)\nmodel = TextToAMR(model_name)")

        from huggingface_hub import snapshot_download
        if "local_dir" not in hf_kwargs:
            hf_kwargs["local_dir"] = f"{root_dir}/models"
        
        snapshot_download(repo_id=repo_id, **hf_kwargs)

        return TextToAMR(model_name, root_dir, **kwargs)
    
    def _prepare_predict_dataset(self, sentences: list[str]):
        data_args = self.data_args
        with open(data_args.test_file, encoding="utf-8", mode="w") as fp:
            for x in sentences:
                json_str = json.dumps({"sent": x, "amr": "", "lang": "id"})
                print(json_str, file=fp)

        raw_datasets = AMRParsingDataSet(
            self.tokenizer,
            data_args,
            self.model_args,
            use_lang_prefix=self.use_prefix
        )
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
    
    def _make_graphs(self, preds):
        graphs: list[penman.Graph] = []
        for idx in range(len(preds)):
            ith_pred = preds[idx]
            ith_pred[0] = self.tokenizer.bos_token_id
            ith_pred = [
                self.tokenizer.eos_token_id if itm == self.tokenizer.amr_eos_token_id else itm
                for itm in ith_pred if itm != self.tokenizer.pad_token_id
            ]

            graph, status, (lin, backr) = self.tokenizer.decode_amr(
                ith_pred, restore_name_ops=False
            )
            assert isinstance(graph, penman.Graph)

            # Is this used?
            graph.status = status
            graph.nodes = lin
            graph.backreferences = backr
            graph.tokens = ith_pred

            graphs.append(graph)

        return graphs
