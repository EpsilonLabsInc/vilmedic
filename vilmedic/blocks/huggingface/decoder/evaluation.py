import tqdm
import torch
import torch.nn as nn
from transformers import BertGenerationDecoder
from transformers import GenerationConfig


def get_special_token_ids(model, tokenizer):
    bos_token_id = model.config.bos_token_id
    eos_token_id = model.config.eos_token_id
    pad_token_id = model.config.pad_token_id
    if None in [bos_token_id, eos_token_id, pad_token_id]:
        bos_token_id = tokenizer.vocab[tokenizer.cls_token]
        eos_token_id = tokenizer.vocab[tokenizer.sep_token]
        pad_token_id = tokenizer.vocab[tokenizer.pad_token]

    return bos_token_id, eos_token_id, pad_token_id


def evaluation(models, config, dl, **kwargs):
    models = [m if not isinstance(m, nn.DataParallel) else m.module for m in models]
    hf_models = [model.dec.decoder for model in models]
    hf_model = hf_models[0]  # one model only

    # Get tokenizer and reference sentences from dataloader
    try:
        ref_str = 'input_ids'
        tokenizer = dl.dataset.tokenizer
        max_len = dl.dataset.tokenizer_max_len
    except AttributeError:
        ref_str = 'decoder_input_ids'
        tokenizer = dl.dataset.tgt_tokenizer
        max_len = dl.dataset.tgt_tokenizer_max_len

    # Get tokens
    bos_token_id, eos_token_id, pad_token_id = get_special_token_ids(hf_model, tokenizer)

    ref_list = []
    hyp_list = []

    generation_args = {
        "bos_token_id": bos_token_id,
        "eos_token_id": eos_token_id,
        "pad_token_id": pad_token_id,
        "num_return_sequences": 1,
        "max_length": max_len,
        "use_cache": True,
    }

    # Conditionally add optional parameters if they are not None
    if config.length_penalty is not None:
        generation_args["length_penalty"] = config.length_penalty
    if config.beam_width is not None:
        generation_args["num_beams"] = config.beam_width

    with torch.no_grad():
        for batch in tqdm.tqdm(dl):
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            batch_size = batch[ref_str].shape[0]

            # Getting encoder infos
            encoder_outputs = []
            encoder_attention_masks = []
            for hf in models:
                encoder_output, encoder_attention_mask = hf.encode(**batch)
                encoder_outputs.append(encoder_output)
                encoder_attention_masks.append(encoder_attention_mask)

            encoder_outputs, encoder_attention_mask = hf.encode(**batch)
            # BertGenerationDecoder.generate

            # lets gooooo
            hyps = hf_model.generate(
                input_ids=torch.ones((batch_size, 1), dtype=torch.long).cuda() * bos_token_id,
                generation_config=GenerationConfig(**generation_args),
                encoder_hidden_states=encoder_outputs,
                encoder_attention_mask=encoder_attention_mask
            )

            refs = batch[ref_str]
            for h, r in zip(hyps, refs):
                hyp_list.append(tokenizer.decode(h, skip_special_tokens=True, clean_up_tokenization_spaces=False))
                ref_list.append(tokenizer.decode(r, skip_special_tokens=True, clean_up_tokenization_spaces=False))
            # break
        return {'refs': ref_list, 'hyps': hyp_list}
