import requests
from datasets import load_dataset
from tokenizers import AddedToken, Tokenizer
from tokenizers.models import Unigram
from transformers import AutoTokenizer, PreTrainedTokenizerFast


def get_token_counts(tokenizer, dataset):
    dataset_token_counts = {
        index: 0 for index in tokenizer.get_vocab().values() if index is not None
    }

    for text in dataset:
        tokens = tokenizer.encode(text).tokens
        encoding = [tokenizer.token_to_id(x) for x in tokens]
        for index in encoding:
            dataset_token_counts[index] += 1

    return dataset_token_counts


def find_rare_tokens(token_counts, threshold_fraction=None, threshold_count=None):
    if threshold_fraction is not None:
        assert (
            threshold_count is None
        ), "Either threshold_fraction or threshold_count must be provided"
        threshold_count = threshold_fraction * len(token_counts)
    else:
        assert (
            threshold_count is not None
        ), "Either threshold_fraction or threshold_count must be provided"

    useless_tokens = {
        token_id for token_id, count in token_counts.items() if count < threshold_count
    }

    return set(useless_tokens)


def find_unk_tokens(tokenizer, dataset):
    unk_tokens = set()
    for text in dataset:
        tokens = tokenizer.encode(text).tokens
        encoding = [tokenizer.token_to_id(x) for x in tokens]
        for token, token_index in zip(tokens, encoding):
            if token_index == None:
                unk_tokens.add(token)

    return unk_tokens


def create_unigram_subtokenizer(
    tokenizer, scores, remove_token_ids, ignore_tokens, unk_token_id
):
    remove_tokens = set()
    for token_id in remove_token_ids:
        if token_id in ignore_tokens:
            continue
        token = tokenizer.decode([token_id])
        remove_tokens.add(token)

    subscores = scores[:]
    for i in reversed(range(len(scores))):
        token, _ = scores[i]
        if len(token) == 1:
            continue
        if token not in remove_tokens:
            continue
        subscores[i][1] = -99

    subscores = [tuple(x) for x in subscores]

    subtokenizer_model = Unigram(vocab=subscores, unk_id=unk_token_id)
    subtokenizer = Tokenizer(subtokenizer_model)

    if tokenizer.pre_tokenizer is not None:
        subtokenizer.pre_tokenizer = tokenizer.pre_tokenizer
    if tokenizer.post_processor is not None:
        subtokenizer.post_processor = tokenizer.post_processor
    if tokenizer.normalizer is not None:
        subtokenizer.normalizer = tokenizer.normalizer
    if tokenizer.decoder is not None:
        subtokenizer.decoder = tokenizer.decoder

    return subtokenizer, subscores


def wrap_unigram_autotokenizer(tokenizer, scores):
    ignore_tokens = {
        tokenizer.unk_token_id: tokenizer.unk_token,
        tokenizer.pad_token_id: tokenizer.pad_token,
        tokenizer.eos_token_id: tokenizer.eos_token,
        tokenizer.bos_token_id: tokenizer.bos_token,
        tokenizer.mask_token_id: tokenizer.mask_token,
        tokenizer.cls_token_id: tokenizer.cls_token,
    }
    ignore_tokens = {v: k for k, v in ignore_tokens.items() if k is not None}

    unk_token_id = tokenizer.unk_token_id
    subtokenizer, scores = create_unigram_subtokenizer(
        tokenizer._tokenizer, scores, set(), ignore_tokens, unk_token_id
    )
    return subtokenizer, ignore_tokens, unk_token_id


def fix_unigram_tokenizer(tokenizer_repo, training_dataset, inference_dataset):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_repo)

    tokenizer_data = requests.get(
        f"https://huggingface.co/{tokenizer_repo}/resolve/main/tokenizer.json?download=true"
    ).json()
    tokenizer_config = requests.get(
        f"https://huggingface.co/{tokenizer_repo}/resolve/main/tokenizer_config.json?download=true"
    ).json()
    scores = tokenizer_data["model"]["vocab"]

    tokenizer, ignore_tokens, unk_token_id = wrap_unigram_autotokenizer(
        tokenizer, scores
    )

    # validate the tokenizer against the datasets
    unk_tokens = find_unk_tokens(tokenizer, training_dataset)
    unk_tokens = unk_tokens.union(find_unk_tokens(tokenizer, inference_dataset))
    if unk_tokens:
        raise Exception(f"UNK token found in training dataset: {unk_tokens}")

    pony_token_counts = get_token_counts(tokenizer, training_dataset)
    generic_token_counts = get_token_counts(tokenizer, inference_dataset)
    underrepresented_tokens = find_rare_tokens(pony_token_counts, threshold_count=50)
    useless_tokens = find_rare_tokens(generic_token_counts, threshold_fraction=0.0001)

    bad_tokens = underrepresented_tokens.union(useless_tokens)
    bad_tokens = bad_tokens - set(ignore_tokens.values())

    subtokenizer, subscores = create_unigram_subtokenizer(
        tokenizer, scores, bad_tokens, ignore_tokens, unk_token_id
    )

    vocab = "\n".join(list(subtokenizer.get_vocab()))
    with open("vocab.model", "w") as f:
        f.write(vocab)

    tokenizer_config["added_tokens_decoder"] = {
        k: AddedToken(**v) for k, v in tokenizer_config["added_tokens_decoder"].items()
    }

    converted_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=subtokenizer,
        **tokenizer_config,
    )

    return converted_tokenizer


if __name__ == "__main__":
    generics_kb = load_dataset(
        "community-datasets/generics_kb", name="generics_kb_best", split="train"
    )
    ponyspeech_dataset = load_dataset("synthbot/pony-speech", split="train")

    pony_graphemes = [x.replace("ñ", "n") for x in ponyspeech_dataset["transcription"]]

    n = 240000
    generic_graphemes = generics_kb.shuffle().select(range(n))["generic_sentence"]

    training_dataset = pony_graphemes
    inference_dataset = generic_graphemes + training_dataset

    clean_tokenizer = fix_unigram_tokenizer(
        "parler-tts/parler-tts-mini-v1", training_dataset, inference_dataset
    )
    clean_tokenizer.save_pretrained("./fixed_tokenizer")
