import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import math
from matplotlib.pyplot import imshow
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from typing import Optional, Union, List
import pickle 

from rich import inspect
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def load_vqa2_json(filename):
    with open(filename, 'r') as f:
        data = [json.loads(line) for line in f.readlines()]
    return data
    
# function for generate embeddings
def generate_features(model, image_processor, conv_mode, tokenizer, image_str: str, question: str, modality: Optional[Union[str, List[str]]] = None) -> dict:
    """Generate features for a single image and question pair.

    Args:
        image_str (str): Base64-encoded image.
        question (str): Question to ask.
        modality (Optional[Union[str, List[str]]], optional): Modality to use. Defaults to both.

    Returns:
        dict: Dictionary of features.
    """

    if modality is None:
        modality = ["text", "image"]
    else:
        modality = [modality] if isinstance(modality, str) else modality

    # load image and preprocess it        
    image = Image.open(image_str)
    processed_image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
    processed_image_tensor = processed_image.unsqueeze(0).half().cuda()

    # generate the image features
    image_features = model.get_model().get_vision_tower().to(device='cuda', dtype=torch.float16)(processed_image_tensor)
    image_features = model.get_model().mm_projector(image_features)

    # tokenize the question
    qs = question
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
    
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    # print(prompt)
    # print(input_ids)
    # print([tokenizer(chunk).input_ids for chunk in prompt.split('<image>')])

    # print(model.get_model().embed_tokens.weight.shape)
    # print(len(tokenizer))
    input_ids = input_ids[input_ids != -200]

    with torch.no_grad():
        emb = model.get_model().embed_tokens
        qs_emb = emb(input_ids)

    # input_ids = input_ids.squeeze()
    # print(input_ids.shape``)
    # embed_tokens = None
    # for name, module in model.get_model().named_modules():
    #     if name == 'embed_tokens':
    #         embed_tokens = module
    #         break
    # qs_emb = embed_tokens(input_ids)

    # # * convert the features to numpy array
    qs_np = qs_emb.squeeze().detach().cpu().numpy()
    # print(qs_np.shape)

    img_np = image_features.squeeze().detach().cpu().numpy()
    # print(img_np.shape)

    res = {}
    if "text" in modality:
        res["question_embeddings"] = qs_np
    else:
        res["question_embeddings"] = None
    if "image" in modality:
        res["image_embeddings"] = img_np
    else:
        res["image_embeddings"] = None
    
    # return the features
    return res


def eval_model(args):
    vqa2_json_file = "/home/yifanyang/vqav2/val/qs/val_vqav2.json"
    vqav2_json_data = load_vqa2_json(vqa2_json_file)

    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    model.eval()
  
    # * loop over the whole json dataset and generate the image and text features and store them in npy files
    # * using rich python library to show the progress bar

    # * create the progress bar
    progress = Progress(
        "[progress.description]{task.description}",
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TextColumn("[cyan]{task.fields[filename]}", justify="right"),
        "[progress.completed]{task.completed}/{task.total}",
        ".",
        TimeRemainingColumn(),
    )

    # * variables to store the features
    image_features = []
    qs_features = {}


    frac = 0.05

    # * generate a random test point
    # test_sample = vqav2_json_data[0]
    # test_image = test_sample["image_dir"] + "/val2014/" + test_sample["image"] 
    # test_question = test_sample["question"] 

    # res = generate_features(model, image_processor, args.conv_mode, tokenizer, test_image, test_question, ["image", "text"])

    # print(res["question_embeddings"].shape if res["question_embeddings"] is not None else None)
    



    for idx, sample in tqdm(enumerate(vqav2_json_data[:int(len(vqav2_json_data) * frac)]), total=int(len(vqav2_json_data) * frac), desc="Generating features"):
        test_sample = sample
        test_image = test_sample["image_dir"] + "/val2014/" + test_sample["image"] 
        test_question = test_sample["question"]
        qs = test_question

        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
        
        # print(prompt)
        # print(input_ids)

        # model.to("cpu")
        # with torch.no_grad():
        #     qs_emb = model.get_model().embed_tokens(input_ids)
        
        # print(model.get_model())
        
        res = generate_features(model, image_processor, args.conv_mode, tokenizer, test_image, test_question, ["image", "text"])

        image_features.append(res["image_embeddings"])
        qs_features[idx] = res["question_embeddings"]

    print(np.array(image_features).shape)
    print(len(qs_features))

    # * save the features as npy files in the same directory
    np.save("vqa2_image_feature.npy", np.array(image_features))
    # * save the question features as pickle file 
    with open("vqa2_question_feature.pkl", "wb") as f:
        pickle.dump(qs_features, f, protocol=pickle.HIGHEST_PROTOCOL)
    





    # questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    # questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    # answers_file = os.path.expanduser(args.answers_file)
    # os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    # ans_file = open(answers_file, "w")
    # for line in tqdm(questions):
    #     idx = line["question_id"]
    #     image_file = line["image"]
    #     qs = line["text"]
    #     cur_prompt = qs
    #     if model.config.mm_use_im_start_end:
    #         qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    #     else:
    #         qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    #     conv = conv_templates[args.conv_mode].copy()
    #     conv.append_message(conv.roles[0], qs)
    #     conv.append_message(conv.roles[1], None)
    #     prompt = conv.get_prompt()

    #     input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    #     image = Image.open(os.path.join(args.image_folder, image_file))
    #     image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

    #     stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    #     keywords = [stop_str]
    #     stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            

        # with torch.inference_mode():
        #     output_ids = model.generate(
        #         input_ids,
        #         images=image_tensor.unsqueeze(0).half().cuda(),
        #         do_sample=True if args.temperature > 0 else False,
        #         temperature=args.temperature,
        #         top_p=args.top_p,
        #         num_beams=args.num_beams,
        #         # no_repeat_ngram_size=3,
        #         max_new_tokens=1024,
        #         use_cache=True)

    #     input_token_len = input_ids.shape[1]
    #     n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    #     if n_diff_input_output > 0:
    #         print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    #     outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    #     outputs = outputs.strip()
    #     if outputs.endswith(stop_str):
    #         outputs = outputs[:-len(stop_str)]
    #     outputs = outputs.strip()

    #     ans_id = shortuuid.uuid()
    #     ans_file.write(json.dumps({"question_id": idx,
    #                                "prompt": cur_prompt,
    #                                "text": outputs,
    #                                "answer_id": ans_id,
    #                                "model_id": model_name,
    #                                "metadata": {}}) + "\n")
    #     ans_file.flush()
    # ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)