from torch import nn
import torch
from attention_store import AttentionStore


def passing_argument(args):
    global do_local_self_attn
    global only_local_self_attn
    global fixed_window_size
    global argument

    argument = args



def register_attention_control(unet: nn.Module,controller: AttentionStore):

    def ca_forward(self, layer_name):

        def forward(hidden_states, context=None, trg_layer_list=None, noise_type=None, **model_kwargs):

            print(f'controlling cross attention')
            is_cross_attention = False
            if context is not None:
                is_cross_attention = True
            """ cross self rechecking necessary """

            if noise_type is not None :
                position_embedder, global_net = noise_type
                if argument.use_position_embedder and not argument.use_global_network :
                    hidden_states = position_embedder(hidden_states, layer_name)

                if argument.use_position_embedder and argument.use_global_network :
                    position_embedder, global_net = noise_type
                    lcoal_hidden_states = position_embedder(hidden_states, layer_name)
                    global_hidden_states = global_net(hidden_states, layer_name)
                    if argument.local_hidden_states_globalize :
                        global_hidden_states = global_net(lcoal_hidden_states, layer_name)
                    hidden_states = lcoal_hidden_states + global_hidden_states

            query = self.to_q(hidden_states)
            context = context if context is not None else hidden_states
            key = self.to_k(context)
            value = self.to_v(context)
            query = self.reshape_heads_to_batch_dim(query)
            key = self.reshape_heads_to_batch_dim(key)
            value = self.reshape_heads_to_batch_dim(value)
            if self.upcast_attention:
                query = query.float()
                key = key.float()
            """ Second Trial """
            if trg_layer_list is not None and layer_name in trg_layer_list :
                controller.save_query((query * self.scale), layer_name) # query = batch, seq_len, dim
                controller.save_key(key, layer_name)
            attention_scores = torch.baddbmm(
                torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
                query, key.transpose(-1, -2),
                beta=0,
                alpha=self.scale,)
            attention_probs = attention_scores.softmax(dim=-1).to(value.dtype)
            if trg_layer_list is not None and layer_name in trg_layer_list :
                trg_attn = attention_probs[:,:,:2]
                controller.save_attn(trg_attn, layer_name)
            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
            hidden_states = self.to_out[0](hidden_states)
            return hidden_states
        return forward

    def register_recr(net_, count, layer_name):
        if net_.__class__.__name__ == 'CrossAttention':
            net_.forward = ca_forward(net_, layer_name)
            return count + 1
        elif hasattr(net_, 'children'):
            for name__, net__ in net_.named_children():
                full_name = f'{layer_name}_{name__}'
                count = register_recr(net__, count, full_name)
        return count

    cross_att_count = 0
    for net in unet.named_children():
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, net[0])
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, net[0])
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, net[0])
    controller.num_att_layers = cross_att_count





















# layer_name mid_block_attentions_0_transformer_blocks_0_attn2
# layer_name down_blocks_1_attentions_0_transformer_blocks_0_attn2
# layer_name down_blocks_2_attentions_1_transformer_blocks_0_attn1
# layer_name down_blocks_2_attentions_1_transformer_blocks_0_attn2
# layer_name down_blocks_2_attentions_1_transformer_blocks_0_attn1
# layer_name down_blocks_0_attentions_0_transformer_blocks_0_attn1
# layer_name down_blocks_1_attentions_1_transformer_blocks_0_attn1
# layer_name down_blocks_1_attentions_1_transformer_blocks_0_attn1
# layer_name down_blocks_2_attentions_1_transformer_blocks_0_attn2
# layer_name down_blocks_0_attentions_0_transformer_blocks_0_attn2
# layer_name down_blocks_1_attentions_1_transformer_blocks_0_attn2
# layer_name down_blocks_1_attentions_1_transformer_blocks_0_attn2
# layer_name up_blocks_1_attentions_0_transformer_blocks_0_attn1
# layer_name mid_block_attentions_0_transformer_blocks_0_attn1
# layer_name down_blocks_0_attentions_1_transformer_blocks_0_attn1
# layer_name down_blocks_2_attentions_0_transformer_blocks_0_attn1
# layer_name up_blocks_1_attentions_0_transformer_blocks_0_attn2
# layer_name mid_block_attentions_0_transformer_blocks_0_attn2
# layer_name down_blocks_2_attentions_0_transformer_blocks_0_attn1
# layer_name down_blocks_0_attentions_1_transformer_blocks_0_attn2
# layer_name mid_block_attentions_0_transformer_blocks_0_attn1
# layer_name down_blocks_2_attentions_0_transformer_blocks_0_attn2
# layer_name down_blocks_2_attentions_0_transformer_blocks_0_attn2
# layer_name mid_block_attentions_0_transformer_blocks_0_attn2
# layer_name up_blocks_1_attentions_1_transformer_blocks_0_attn1
# layer_name down_blocks_2_attentions_1_transformer_blocks_0_attn1
# layer_name up_blocks_1_attentions_1_transformer_blocks_0_attn2
# layer_name down_blocks_1_attentions_0_transformer_blocks_0_attn1
# layer_name down_blocks_2_attentions_1_transformer_blocks_0_attn1
# layer_name down_blocks_2_attentions_1_transformer_blocks_0_attn2
# layer_name down_blocks_1_attentions_0_transformer_blocks_0_attn2
# layer_name down_blocks_2_attentions_1_transformer_blocks_0_attn2
# layer_name up_blocks_1_attentions_2_transformer_blocks_0_attn1
# layer_name up_blocks_1_attentions_0_transformer_blocks_0_attn1
# layer_name up_blocks_1_attentions_2_transformer_blocks_0_attn2