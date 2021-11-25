# IJCAI-2021 HITA-Graph

Paper resources for 《Knowledge-Aware Dialogue Generation via Hierarchical Infobox Accessing and Infobox-Dialogue Interaction Graph Network》. IJCAI 2021.

The released version may have some minor differences. 

# Dataset

Please download it from [Google Drive](https://drive.google.com/file/d/1-sxVKXHd5y68kPJQSAyxESzN3lHuXezn/view?usp=sharing)

Note:  Four files, `filed_key.txt.char....`, were missed in the above dataset link. We have updated the code (see `dataset/table_dialogue_res/...`). Please pull the project.
# Requirements
```
Python=3.6
PyTorch==1.4.0
TorchText==0.6.0
```


#Training

To train our model, you can use the following script (or you can find this in `scripts.sh`):
```shell script
 python -u -m table2seq.run_t2s -bridge fusion  -train_data_path_prefix dataset/table_dialogue_res/train -val_data_path_prefix dataset/table_dialogue_res/dev -test_data_path_prefix dataset/table_dialogue_res/test -vocab_path dataset/table_dialogue_res/vocab.txt -field_key_vocab_path dataset/table_dialogue_res/field_key.txt  -mode train -cuda True -hidden_size 512 -embed_size 200 -batch_size 50 -epochs 30 -enc_layers 1 -dec_layers 1 -dual_attn general -field_word_vocab_path dataset/table_dialogue_res/field_word.txt -model_path models/ijcai_2021/HitaGraph -lr 0.0001 -field_tag_usage general  -field_word_tag_path dataset/table_dialogue_res/field_tag.txt  -src_tag_vocab_path dataset/table_dialogue_res/pos_tag.txt   -field_encoder hierarchical_infobox  -enable_field_attn True -enable_query_attn True -copy True -field_copy True -share_copy_attn True -share_field_copy_attn True -infobox_memory_bank_format fwk_fwv_fk -field_input_tags local_pos_fw,local_pos_bw  -mode_selector mlp -add_last_generated_token True  -init_word_vecs True  -unk_learning skip -hierarchical_infobox_rgat_layers 2  -update_decoder_with_global_node True -add_state_to_copy_token True  -copy_query_first True  -char_encoder_type cnn -char_encoders field_key,field_word -hierarchical_infobox_attention True -hierarchical_infobox_attention_type general
```
The given script uses the data stored in  `dataset/table_dialogue_res/`, and stores checkpoints in `-model_path models/ijcai_2021/HitaGraph`, please change such parameters before running our model.

We load the embedding from `--pre_embed_file`, you need to set this parameter as well.  We use the embedding released by [Tencent AI](https://ai.tencent.com/ailab/nlp/en/data/Tencent_AILab_ChineseEmbedding.tar.gz) 
 
#Inference 
You can use the following script to infer the queries given by the  `test set`.

You can change `-beam_width` to meet your requirements.

```shell script
python -u -m table2seq.run_t2s  -bridge fusion   -train_data_path_prefix dataset/table_dialogue_res/train -val_data_path_prefix dataset/table_dialogue_res/dev -test_data_path_prefix dataset/table_dialogue_res/test -vocab_path dataset/table_dialogue_res/vocab.txt -field_key_vocab_path dataset/table_dialogue_res/field_key.txt  -mode train -cuda True -hidden_size 512 -embed_size 200 -batch_size 32 -epochs 30 -enc_layers 1 -dec_layers 1 -dual_attn general -field_word_vocab_path dataset/table_dialogue_res/field_word.txt -model_path models/ijcai_2021/HitaGraph -lr 0.0001  -field_tag_usage general  -field_word_tag_path dataset/table_dialogue_res/field_tag.txt  -src_tag_vocab_path dataset/table_dialogue_res/pos_tag.txt   -field_encoder hierarchical_infobox    -mode_selector mlp -add_last_generated_token True  -init_word_vecs True  -unk_learning skip  -enable_field_attn True -enable_query_attn True -copy True -field_copy True   -share_copy_attn True -share_field_copy_attn True   -infobox_memory_bank_format fwk_fwv_fk  -field_input_tags local_pos_fw,local_pos_bw -mode infer -beam_width 10  -hierarchical_infobox_rgat_layers 2  -update_decoder_with_global_node True -add_state_to_copy_token True -copy_query_first True  -char_encoder_type cnn -char_encoders field_key,field_word -hierarchical_infobox_attention True -hierarchical_infobox_attention_type general &
```

# Citation
```
@inproceedings{DBLP:conf/ijcai/WuWZZ0W21,
  author    = {Sixing Wu and
               Minghui Wang and
               Dawei Zhang and
               Yang Zhou and
               Ying Li and
               Zhonghai Wu},
  editor    = {Zhi{-}Hua Zhou},
  title     = {Knowledge-Aware Dialogue Generation via Hierarchical Infobox Accessing
               and Infobox-Dialogue Interaction Graph Network},
  booktitle = {Proceedings of the Thirtieth International Joint Conference on Artificial
               Intelligence, {IJCAI} 2021, Virtual Event / Montreal, Canada, 19-27
               August 2021},
  pages     = {3964--3970},
  publisher = {ijcai.org},
  year      = {2021},
  url       = {https://doi.org/10.24963/ijcai.2021/546},
  doi       = {10.24963/ijcai.2021/546},
  timestamp = {Wed, 25 Aug 2021 17:11:16 +0200},
  biburl    = {https://dblp.org/rec/conf/ijcai/WuWZZ0W21.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
