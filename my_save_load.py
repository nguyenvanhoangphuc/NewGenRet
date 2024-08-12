from copy import deepcopy
import torch
import os
from my_prepare_data import BiDataset


def safe_load(model, file):
    state_dict = torch.load(file, map_location=lambda storage, loc: storage)
    model_state_dict_keys = list(model.state_dict().keys())
    new_state_dict_keys = list(state_dict.keys())
    new_keys_in_new = [k for k in new_state_dict_keys if k not in model_state_dict_keys]
    no_match_keys_of_model = [k for k in model_state_dict_keys if k not in new_state_dict_keys]
    # size_not_match = [k for k,v in state_dict.items() if model_state_dict_keys[k]]
    print('##', model._get_name(), '# new keys in file:', new_keys_in_new, '# no match keys:', no_match_keys_of_model)
    model.load_state_dict(state_dict, strict=False)


def safe_load_embedding(model, file):
    state_dict = torch.load(file, map_location=lambda storage, loc: storage)
    model_state_dict_keys = list(model.state_dict().keys())
    new_state_dict_keys = list(state_dict.keys())
    new_keys_in_new = [k for k in new_state_dict_keys if k not in model_state_dict_keys]
    no_match_keys_of_model = [k for k in model_state_dict_keys if k not in new_state_dict_keys]
    print('##', model._get_name(), '# new keys in file:', new_keys_in_new, '# no match keys:', no_match_keys_of_model)

    matched_state_dict = deepcopy(model.state_dict())
    for key in model_state_dict_keys:
        if key in state_dict:
            file_size = state_dict[key].size(0)
            model_embedding = matched_state_dict[key].clone()
            model_size = model_embedding.size(0)
            model_embedding[:file_size, :] = state_dict[key][:model_size, :]
            matched_state_dict[key] = model_embedding
            print(f'Copy {key} {matched_state_dict[key].size()} from {state_dict[key].size()}')
    model.load_state_dict(matched_state_dict, strict=False)


def safe_save(accelerator, model, save_path, epoch, end_epoch=100, save_step=9, last_checkpoint=None):
    os.makedirs(save_path, exist_ok=True)
    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process and epoch < end_epoch and epoch % save_step == 0:
        unwrap_model = accelerator.unwrap_model(model)
        accelerator.save(unwrap_model.state_dict(), f'{save_path}/{epoch}.pt')
        accelerator.save(unwrap_model.model.state_dict(), f'{save_path}/{epoch}.pt.model')
        accelerator.save(unwrap_model.centroids.state_dict(), f'{save_path}/{epoch}.pt.centroids')
        accelerator.save(unwrap_model.code_embedding.state_dict(), f'{save_path}/{epoch}.pt.embedding')
        accelerator.print(f'Save model {save_path}/{epoch}.pt')
        last_checkpoint = f'{save_path}/{epoch}.pt'
    return epoch + 1, last_checkpoint


def simple_loader(data, corpus, tokenizer, ids, aux_ids, accelerator):
    dataset = BiDataset(data=data, corpus=corpus, tokenizer=tokenizer,
                        max_doc_len=128, max_q_len=32, ids=ids, batch_size=1, aux_ids=aux_ids)
    data_loader = torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn, batch_size=32,
                                              shuffle=True, num_workers=4)
    data_loader = accelerator.prepare(data_loader)
    return data_loader