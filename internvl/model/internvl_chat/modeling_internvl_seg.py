from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from internvl.conversation import get_conv_template
from internvl.model.internlm2.modeling_internlm2 import InternLM2ForCausalLM
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import logging

from .configuration_internvl_chat import InternVLChatConfig
from .modeling_internvl_chat import InternVLChatModel
from ...model.var_vae.models import VQVAE, build_vae
# from ...model.var_vae.models.helpers import sample_with_top_k_top_p_

logger = logging.get_logger(__name__)

@dataclass
class CustomCausalLMOutputWithPast(CausalLMOutputWithPast):
    text_loss: float = None
    gen_loss: float = None

def version_cmp(v1, v2, op='eq'):
    import operator

    from packaging import version
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))


class InternLM2ForCausalLM_Seg(InternLM2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.image_tokenizer = None
        self.gen_cache = []
        self.cfg = 4
        self.return_multi_results = False
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if self.gen_cache:
            img_num = len(self.gen_cache)
            new_attention = torch.ones((attention_mask.shape[0], 680*img_num), dtype=attention_mask.dtype, device=attention_mask.device)
            position_ids = position_ids[:,-1][:,None] + 680*img_num
            attention_mask = torch.cat([attention_mask, new_attention], dim=-1)
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds = inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        logits = outputs.logits
        loss = outputs.loss
        # infer mode 
        if not self.training:
            predicted_token_ids = torch.argmax(logits.clone().float(), dim=-1)[0, -1]
            if predicted_token_ids == self.gen_start_token_id:
                outputs = self.decode_var(logits, use_cache, outputs.past_key_values, attention_mask, position_ids)

                new_logits = logits.new_full(logits.shape, -1000.0)  
                new_logits[..., self.gen_end_token_id ] = 1000.0 
                logits = new_logits # set new predict <gen_end>
        
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        output['logits'] = output['logits'].to(device)
        return output
        
    def decode_var(self, logits, use_cache, past_key_values, attention_mask=None, position_ids=0):
        import PIL.Image as PImage
        import numpy as np        

        predicted_token_ids = torch.argmax(logits, dim=-1) 
        # ge_start_id = self.tokenizer.convert_tokens_to_ids('<ge_start>')

        cur_L = 0
        B = logits.shape[0]
        sog = self.get_input_embeddings()(predicted_token_ids).clone()
        next_token_map = sog.expand(B, self.patch_nums[0] ** 2, -1)
        self.Cvae = self.image_tokenizer.Cvae
        f_hat = logits.new_zeros(1, self.image_tokenizer.Cvae, self.patch_nums[-1], self.patch_nums[-1]) # [8, 32, 16, 16]
        lvl_pos = self.lvl_embed(self.lvl_1L.to(self.pos_1LC.device)) + self.pos_1LC # Embedding(10, 1536) level embedding  [680,1536]
        
        visual_ids_pre = []
        for si, pn in enumerate(self.patch_nums):   # si: i-th segment
            cur_L += pn*pn

            inputs_embeds = next_token_map
            position_ids = position_ids[:,-1][:,None] + 1
            for num_i in range(self.patch_nums[si] ** 2-1):
                position_ids = torch.cat((position_ids, position_ids[:,-1][:,None] + 1), dim=-1)
            new_attention = torch.ones((B, self.patch_nums[si] ** 2), dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([attention_mask, new_attention], dim=-1)
            outputs = self.model(
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values

            hidden_states = outputs[0]
            logits_BlV = self.output(hidden_states)
            logits_BlV = logits_BlV[:,:,-4096:]
            if B > 1:
                cfg = self.cfg # 4
                ratio = si/9 # 1
                t = cfg * ratio
                logits_BlV = (1+t) * logits_BlV[:1] - t * logits_BlV[1:]
                # logits_BlV = t * logits_BlV[:1] + (1-t) * logits_BlV[1:]
            idx_Bl = torch.argmax(logits_BlV, dim=-1)
            # idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=None, top_k=900, top_p=0.96, num_samples=1)[:, :, 0]

            visual_ids_pre.append(idx_Bl)
            
            h_BChw = self.image_tokenizer.quantize.embedding(idx_Bl)   # B, l, Cvae
            
            h_BChw = h_BChw.transpose_(1, 2).reshape(1, self.Cvae, pn, pn)
            f_hat, next_token_map = self.image_tokenizer.quantize.get_next_autoregressive_input(si, len(self.patch_nums), f_hat, h_BChw)
            if si != (len(self.patch_nums) - 1):   # prepare for next stage
                next_token_map = next_token_map.view(1, self.Cvae, -1).transpose(1, 2)
                next_token_map = self.word_embed(next_token_map.to(logits_BlV.dtype)) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
                if B != 1:
                    next_token_map = next_token_map.repeat(2, 1, 1)  # double the batch sizes due to CFG
        ################################################################################
        visual_ids_pre = torch.cat(visual_ids_pre,dim=-1)
        visual_ids_pre = self.to_image_tokens(visual_ids_pre) # [0]

        img_pre = self.image_tokenizer.decode(visual_ids_pre, same_shape=True)[-1]

        if self.return_multi_results:
            tmp = self.image_tokenizer.decode(visual_ids_pre, same_shape=True)
            self.multi_mask = []
            for i in range(len(tmp)):
                self.multi_mask.append((tmp[i][0].clone().permute(1, 2, 0).add_(1).mul_(0.5).mul_(255).cpu().float().numpy()).astype(np.uint8))

        img_pre =  PImage.fromarray((img_pre[0].permute(1, 2, 0).add_(1).mul_(0.5).mul_(255).cpu().float().numpy()).astype(np.uint8))
        self.gen_cache.append(img_pre)
        # self.image_num  += 1
        return outputs

    def to_image_tokens(self, image_docs):

        token_ids = image_docs

        # get visual_decode input
        v_patch_nums=[pn**2 for pn in self.patch_nums]
        cumsum = [0] + [sum(v_patch_nums[:i+1]) for i in range(len(v_patch_nums))]
        # token_ids = [token_ids[cumsum[i]:cumsum[i+1]][None,:] for i in range(len(cumsum)-1)]
        token_ids = [token_ids[:, cumsum[i]:cumsum[i+1]] for i in range(len(cumsum)-1)]

        return token_ids
    
    
class InternVLGenSeg(InternVLChatModel):
    gen_img_reso = 256
    init_vae = True

    def __init__(self, config: InternVLChatConfig, vision_model=None, language_model=None, use_flash_attn=True):
        super().__init__(config)

        self.language_model = InternLM2ForCausalLM_Seg(config.llm_config)
        #################################################################
        if InternVLGenSeg.gen_img_reso == 256:
            self.patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
        else:
            self.patch_nums = (1, 2, 3, 4, 6, 9, 13, 18, 24, 32)
        #### the following is new weights
        def reset_parameters(linear_layer):
            nn.init.zeros_(linear_layer.weight)
            if linear_layer.bias is not None:
                nn.init.zeros_(linear_layer.bias)

        llm_hidden_size = config.llm_config.hidden_size
        self.llm_hidden_size = llm_hidden_size

        self.word_embed = nn.Linear(32, llm_hidden_size)
        reset_parameters(self.word_embed)

        import math
        init_std = math.sqrt(1 / llm_hidden_size / 3)
        pos_1LC = []
        self.L = sum(pn ** 2 for pn in self.patch_nums)
        for i, pn in enumerate(self.patch_nums):
            pe = torch.empty(1, pn*pn, llm_hidden_size)
            nn.init.zeros_(pe)
            pos_1LC.append(pe)
        pos_1LC = torch.cat(pos_1LC, dim=1)     # 1, L, C
        assert tuple(pos_1LC.shape) == (1, self.L, llm_hidden_size)
        self.pos_1LC = nn.Parameter(pos_1LC)
        # level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
        self.lvl_embed = nn.Embedding(len(self.patch_nums), llm_hidden_size)
        nn.init.zeros_(self.lvl_embed.weight.data)

        d: torch.Tensor = torch.cat([torch.full((pn*pn,), i) for i, pn in enumerate(self.patch_nums)]).view(1, self.L, 1)
        dT = d.transpose(1, 2)    # dT: 11L
        self.lvl_1L = dT[:, 0].contiguous()

        self.image_num = 0
        self.reconstruct_tokens_num = 680 if self.patch_nums[-1] == 16 else 2240
        self.gen_cache = None
        logger.info(f'num_mask_token: {self.reconstruct_tokens_num}')
        self.mask_context_token_id = 92556 # None
        self.gen_start_token_id = 92554

        patch_nums=self.patch_nums
        if InternVLGenSeg.init_vae:
            vae = build_vae(
                V=4096, Cvae=32, ch=160, share_quant_resi=4,    # hard-coded VQVAE hyperparameters
                device="cpu",
                patch_nums = patch_nums, load_pretrained = False
            )
            self.set_visual_tokenizer(vae)
        #################################################################

        # Initialize loss tracking for logging
        self.text_loss = None
        self.gen_loss = None

    def init_seg_weights(self):
        return
        self.word_embed.reset_parameters()

        import math
        llm_hidden_size = self.llm_hidden_size
        init_std = math.sqrt(1 / llm_hidden_size / 3)
        pos_1LC = []
        self.L = sum(pn ** 2 for pn in self.patch_nums)
        for i, pn in enumerate(self.patch_nums):
            pe = torch.empty(1, pn*pn, llm_hidden_size)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        pos_1LC = torch.cat(pos_1LC, dim=1)     # 1, L, C
        assert tuple(pos_1LC.shape) == (1, self.L, llm_hidden_size)
        self.pos_1LC = nn.Parameter(pos_1LC)
        # level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        print('Init Seg Weights Done!')


    def forward(
            self,
            pixel_values: torch.FloatTensor,
            masks: torch.FloatTensor = None,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            mask_flags: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            statistics: Optional[torch.LongTensor] = None,
            loss_weight: Optional[List] = None,
            loss_reduction_all_gather: Optional[bool] = False,
    ) -> Union[Tuple, CustomCausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_flags = image_flags.squeeze(-1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

        vit_embeds = self.extract_feature(pixel_values)
        vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            print(f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')
            if statistics is not None:
                num_samples, num_padding_tokens, num_padding_images = statistics.tolist()
                self.num_samples += num_samples
                print(f'total_samples={self.num_samples}, {num_samples=}, {num_padding_tokens=}, {num_padding_images=}')

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)

        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
            ignore_flag = False
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                  f'vit_embeds.shape={vit_embeds.shape}')
            n_token = selected.sum()
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]
            ignore_flag = True        
        
        ###############################################################
        if masks is not None:
            with torch.no_grad(): 
                mask_token_ids = self.image_tokenizer.encode(masks)
                visual_embed = self.image_tokenizer.quantize.idxBl_to_var_input(mask_token_ids)
                mask_token_ids_llm = torch.cat(mask_token_ids,dim=-1) + self.mask_context_token_id + 1
            visual_embed_llm = self.word_embed(visual_embed.to(masks.dtype))
            VAE_batch_size = visual_embed_llm.shape[0]
            visual_embed_llm += self.lvl_embed(self.lvl_1L[:, 1:self.reconstruct_tokens_num].expand(VAE_batch_size, -1).to(self.pos_1LC.device)) + self.pos_1LC[:, 1:self.reconstruct_tokens_num]
            mask_flags = mask_flags.squeeze(-1)
            visual_embed_llm = visual_embed_llm[mask_flags == 1]
            mask_token_ids_llm = mask_token_ids_llm[mask_flags == 1]

        selected_mask = (input_ids == self.mask_context_token_id)
        try:
            input_embeds[selected_mask] = input_embeds[selected_mask] * 0.0 + visual_embed_llm.reshape(-1, C)
            labels = labels.reshape(B * N)
            label_selected_mask = (labels == self.mask_context_token_id)
            labels[label_selected_mask] = mask_token_ids_llm.reshape(-1)
            labels = labels.reshape(B, N)
            ignore_flag = False
        except Exception as e:
            visual_embed_llm = visual_embed_llm.reshape(-1, C)
            print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected_mask].shape}, '
                  f'visual_embed_llm.shape={visual_embed_llm.shape}')
            n_token = selected_mask.sum()
            input_embeds[selected_mask] = input_embeds[selected_mask] * 0.0 + visual_embed_llm[:n_token]
            ignore_flag = True
        ###############################################################

        input_embeds = input_embeds.reshape(B, N, C)

        output_hidden_states = False
        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        loss = None
        if labels is not None and loss_weight is not None:
            loss_weight = torch.tensor(loss_weight, dtype=torch.float32, device=labels.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_weights = loss_weight[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction='none')
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_weights = shift_weights.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            shift_weights = shift_weights.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

            shift_weights_sum = shift_weights.sum()
            if loss_reduction_all_gather:
                dist.all_reduce(shift_weights_sum, op=dist.ReduceOp.AVG)

            loss = loss * shift_weights
            loss = loss.sum() / shift_weights_sum
            if ignore_flag:
                loss = loss * 0.0
        elif labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            text_loss = loss_fct(shift_logits[shift_labels<=self.mask_context_token_id], shift_labels[shift_labels<=self.mask_context_token_id])
            if selected_mask.sum() > 0:
                gen_loss = loss_fct(shift_logits[shift_labels>self.mask_context_token_id], shift_labels[shift_labels>self.mask_context_token_id])
                # loss = loss_fct(shift_logits, shift_labels)
                loss = text_loss + gen_loss
            else:
                gen_loss = None
                loss = text_loss
            if ignore_flag:
                loss = loss * 0.0

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        # Store losses for logging callback
        self.text_loss = text_loss
        self.gen_loss = gen_loss

        return CustomCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            text_loss=text_loss,
            gen_loss=gen_loss,
        )
    
    # @torch.no_grad()
    def set_visual_tokenizer(self, image_tokenizer):
        self.image_tokenizer: VQVAE = image_tokenizer
        self.image_tokenizer.eval()

    def batch_chat(self, tokenizer, pixel_values, questions, generation_config, num_patches_list=None,
                   history=None, return_history=False, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>',
                   DEFAULT_GE_START_TOKEN='<ge_start>', DEFAULT_GE_END_TOKEN='<ge_end>', MASK_CONTEXT_TOKEN='<MASK_CONTEXT>', 
                   IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', verbose=False, image_counts=None, cfg=4.0):
        if history is not None or return_history:
            print('Now multi-turn chat is not supported in batch_chat.')
            raise NotImplementedError

        if image_counts is not None:
            num_patches_list = image_counts
            print('Warning: `image_counts` is deprecated. Please use `num_patches_list` instead.')

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        gen_start_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_GE_START_TOKEN)
        gen_end_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_GE_END_TOKEN)
        self.gen_start_token_id = gen_start_token_id
        self.gen_end_token_id = gen_end_token_id
        template = get_conv_template(self.template)
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
        self.language_model.eos_token_id = eos_token_id
        self.language_model.cfg = cfg
        self.language_model.tokenizer = tokenizer

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        queries = []
        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else [0]*len(questions)
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and '<image>' not in question:
                question = '<image>\n' + question
            template = get_conv_template(self.template)
            template.system_message = self.system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
            queries.append(query)

        tokenizer.padding_side = 'left'
        model_inputs = tokenizer(queries, return_tensors='pt', padding=True)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_ids = model_inputs['input_ids'].to(device)
        attention_mask = model_inputs['attention_mask'].to(device)
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        responses = [response.split(template.sep.strip())[0].strip() for response in responses]
        return responses
    
    def chat(self, tokenizer, pixel_values, question, generation_config, history=None, return_history=False,
             num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
             DEFAULT_GE_START_TOKEN='<ge_start>', DEFAULT_GE_END_TOKEN='<ge_end>', MASK_CONTEXT_TOKEN='<MASK_CONTEXT>', 
             return_multi_results=False, verbose=False):

        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        gen_start_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_GE_START_TOKEN)
        gen_end_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_GE_END_TOKEN)
        self.gen_start_token_id = gen_start_token_id
        self.gen_end_token_id = gen_end_token_id

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
        self.language_model.eos_token_id = eos_token_id
        self.language_model.return_multi_results = return_multi_results

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors='pt')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_ids = model_inputs['input_ids'].to(device)
        attention_mask = model_inputs['attention_mask'].to(device)
        generation_config['eos_token_id'] = eos_token_id
        self.language_model.gen_cache = []
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )

        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(template.sep.strip())[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            if verbose:
                print(query_to_print, response)
            return response

    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:

        assert self.img_context_token_id is not None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.extract_feature(pixel_values)
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        ##### set language model attributes #####
        self.language_model.image_tokenizer = self.image_tokenizer
        self.language_model.gen_start_token_id = self.gen_start_token_id
        self.language_model.gen_end_token_id = self.gen_end_token_id
        self.language_model.mask_context_token_id = self.mask_context_token_id
        self.language_model.lvl_embed = self.lvl_embed
        self.language_model.patch_nums = self.patch_nums
        self.language_model.lvl_1L = self.lvl_1L
        self.language_model.pos_1LC = self.pos_1LC
        self.language_model.word_embed = self.word_embed
        self.language_model.image_num = 0

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs
