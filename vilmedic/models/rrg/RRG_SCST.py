import torch.nn as nn
from vilmedic.models.utils import get_n_params
import functools


from vilmedic.blocks.vision import *
from vilmedic.blocks.huggingface.decoder.evaluation import evaluation as evaluation_

from .RRG import RRG
import numpy as np
import copy
import torch
import logging
import torch.nn.functional as F
import inspect
import random
from vilmedic.blocks.scorers.scores import REWARD_COMPLIANT


def evaluation(models, config, dl, **kwargs):
    models = [m.model for m in models]  # Get trained RRG instance
    return evaluation_(models, config, dl)


def scst_loss(input, seq, reward, pad_token_id):
    # HuggingFace TopKLogitsWarper (if top_k > 0) puts -float('inf') for non top_k (or if we use bad_words_ids)
    # Padding can then have logits -float('inf') though we have to pad because hyp generation is finished
    # -float('inf') * 0 masking results in NaN so that doesnt mitigate the issue. need to do:
    input[input == -float("Inf")] = 0.
    #####

    # reward
    N, L = input.shape[:2]
    reward = reward.view(N, 1, 1).expand_as(input)
    input = -input * reward

    # masking
    mask = (seq > pad_token_id).float()
    output = input.view(-1) * mask.view(-1)

    # mean
    output = torch.sum(output) / torch.sum(mask)
    return output


class RRG_SCST(nn.Module):

    def __init__(self, decoder, cnn, ckpt, dl, logger=None, score="ROUGEL", score_args=None, top_k=None, **kwargs):
        super().__init__()
        if score_args is None:
            score_args = {}

        # Models
        state_dict = torch.load(ckpt)["model"]
        self.model = RRG(copy.deepcopy(decoder), copy.deepcopy(cnn), **kwargs)
        self.model.load_state_dict(state_dict, strict=True)

        self.top_k = top_k
        assert score in REWARD_COMPLIANT
        self.scorer = REWARD_COMPLIANT[score](**score_args)

        # Tokens
        self.bos_token_id = self.model.dec.decoder.config.bos_token_id
        self.eos_token_id = self.model.dec.decoder.config.eos_token_id
        self.pad_token_id = self.model.dec.decoder.config.pad_token_id

        # Tokenizer
        self.dl = dl
        self.tokenizer = dl.dataset.tokenizer

        self.eval_func = evaluation
        self.logger = logger or logging.get_logger(__name__)

    def forward(self, input_ids, attention_mask, images, encoder_outputs=None, **kwargs):
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]

        # 1 Greedy
        with torch.no_grad():
            self.model.eval()
            encoder_hidden_states, encoder_attention_mask = self.model.encode(images.cuda())
            out = self.model.dec.decoder.generate(
                input_ids=torch.ones((batch_size, 1), dtype=torch.long).cuda() * self.bos_token_id,
                attention_mask=attention_mask,
                max_length=seq_len,
                num_beams=1,
                num_return_sequences=1,
                return_dict_in_generate=True,
                output_scores=True,
                encoder_hidden_states=encoder_hidden_states.detach(),
                encoder_attention_mask=encoder_attention_mask.detach(),
                forced_eos_token_id=True,
            )
            greedy_input_ids = out.sequences
            score_greedy = self.get_reward(greedy_input_ids.detach().data, input_ids)

        # 2. Sampling
        self.model.train()
        encoder_hidden_states, encoder_attention_mask = self.model.encode(images.cuda())
        out = inspect.unwrap(self.model.dec.decoder.generate)(  # inspect.unwrap removes the torch.no_grad() decorator
            self=self.model.dec.decoder,
            input_ids=torch.ones((batch_size, 1), dtype=torch.long).cuda() * self.bos_token_id,
            max_length=seq_len,
            num_beams=1,
            num_return_sequences=1,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            bad_words_ids=[[self.pad_token_id], [self.bos_token_id]],
            top_k=self.top_k,
            forced_eos_token_id=True,
            output_scores=True,
            do_sample=True,
            use_cache=True,
            return_dict_in_generate=True,
        )
        samples_ids = out.sequences[:, 1:].contiguous()
        logits = torch.stack(out.scores, dim=1)
        logits = F.log_softmax(logits, dim=-1)
        sampled_logits = logits.gather(2, samples_ids.unsqueeze(-1))

        # 3. Reward and loss
        score_sampling = self.get_reward(samples_ids.data, input_ids)
        reward = torch.tensor(score_sampling).cuda() - torch.tensor(score_greedy).cuda()
        loss = scst_loss(sampled_logits, samples_ids.data, reward, self.pad_token_id)
        # https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/7
        return {"loss": loss, "custom_print": "score_sampling {}".format(np.mean(score_sampling))}

    def get_reward(self, rollout_input_ids, input_ids):

        hyp_list = []
        ref_list = []
        for h, r in zip(rollout_input_ids, input_ids):
            hyp_list.append(self.tokenizer.decode(h, skip_special_tokens=True, clean_up_tokenization_spaces=False))
            ref_list.append(self.tokenizer.decode(r, skip_special_tokens=True, clean_up_tokenization_spaces=False))
        reward = self.scorer(ref_list, hyp_list)
        if isinstance(reward, list) or isinstance(reward, tuple):
            reward = reward[1]
        return reward

    def __repr__(self):
        s = "RRG_PPO\n"
        s += str(self.scorer) + '\n'
        s += "{}\n".format(get_n_params(self))
        return s