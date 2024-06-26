import os
import re

import torch
import transformers
from torch.utils.data import DataLoader
from tqdm import tqdm

from MCCWS.dataset import CWSDataset


def preprocess(data, tokenizer: transformers.PreTrainedTokenizerFast):
    """This function is for tokenization.

    Args:
        data (List[Tuple]): The data fetched from dataset. There are three elements \
                            in each tuple. The first one is the text with spaces    \
                            between each character. The second one is the original  \
                            text to tokenized. The last one idicates whether that   \
                            this input text is finished or not. 
                            (Some data are longer than the model's max length, so we\
                            need to split the kind of data into two pieces.) 
                             
        tokenizer (transformers.PreTrainedTokenizerFast): tokenizer

    Returns:
        tokenized_str : tokenizied input data.
        text : the original text
        short_input :
    """
    tokenized_str = tokenizer(
        [x[0] for x in data],
        truncation=True,
        max_length=512,
        padding="longest",
        return_tensors="pt",
    )

    text, short_input = [x[1] for x in data], [x[2] for x in data]

    return tokenized_str, text, short_input


def valid(
    model,
    tokenizer: transformers.PreTrainedTokenizerFast,
    exp_file: str,
    step: int,
    device: str,
) -> float:
    """In this funciton, we first postprocess the testing data. The word that transformed \
        to 'a' or '0' should be recovered. Then we use the scoring scripts from sighan to \
        evaluate our model's performance.

    Args:
        exp_file (str): The file to store the evaluation result.
        criteria (str): Which dataset we are evaulating.
        step (int): The step you store your model. We store each model with a step, so we \
            need this information to load a specific model to inference.
        device (str): The device use for evaluation.

    Returns:
        F1 score (flaot) : The F1 score result. 
    """
    assert "cuda" in device or "cpu" in device

    try:
        os.mkdir("./exp/result/" + exp_file)
    except:
        pass

    F1 = re.compile(r"=== F MEASURE:\s+0.(\d+)")

    valid_dataset = {
        "[AS]": ["./data/trainset/as_valid.utf8"],
        "[CIT]": ["./data/trainset/cityu_valid.utf8"],
        "[MSR]": ["./data/trainset/msr_valid.utf8"],
        "[PKU]": ["./data/trainset/pku_valid.utf8"],
        "[SXU]": ["./data/testset/sxu_test.txt"],
        "[CTB6]": ["./data/testset/ctb6_test_traditional.txt"],
        "[CNC]": ["./data/testset/cnc_test.txt"],
        "[UD]": ["./data/testset/ud_test.txt"],
        "[ZX]": ["./data/testset/zx_test.txt"],
    }
    for dataset_name, dataset_path in valid_dataset.items():
        dataset = CWSDataset(
            datasets={dataset_name: dataset_path},
            train_set=False,
            criterion_token=dataset_name,
        )
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=256,
            shuffle=False,
            collate_fn=lambda x: preprocess(data=x, tokenizer=tokenizer),
        )

        with torch.no_grad():
            f = open(f"./exp/result/{exp_file}/valid.txt", "w")
            for data, text, short_input in tqdm(data_loader):
                data = data.to(device)
                model_cls, criteria, seg_predict = model.inference(data)
                seg_predict = seg_predict.argmax(dim=-1)
                for i in range(len(seg_predict)):
                    tmp = []
                    ans = ""
                    space_counter = 0
                    for j in range(len(text[i])):
                        if text[i][j] == " ":
                            tmp.append(ans)
                            ans = ""
                            space_counter += 1
                            continue
                        if seg_predict[i][j - space_counter] < 2:
                            if seg_predict[i][j - space_counter] == 0 and len(ans) > 0:
                                tmp.append(
                                    ans
                                )  # This is need since model can wrongly produce sequences like "MB".
                                ans = ""
                            ans += text[i][j]
                        elif seg_predict[i][j - space_counter] == 2:
                            tmp.append(ans + text[i][j])
                            ans = ""
                        else:
                            tmp.append(
                                ans
                            )  # This is need since model can wrongly produce sequences like "BMS".
                            tmp.append(text[i][j])
                            ans = ""
                    if len(ans) > 0:
                        tmp.append(ans)
                    if short_input[i] and tmp[-1][-1] != "\n":
                        tmp.append("\n")
                    f.write(" ".join(tmp))
            f.close()
            os.system(
                f"python3 -m MCCWS.script.postprocess \
                --original_test_gold_path ./icwb2-data/gold/{dataset_name[1:-1].lower()}_valid.utf8 \
                --test_file ./exp/result/{exp_file}/valid.txt"
            )
            os.system(
                f"perl icwb2-data/scripts/score icwb2-data/gold/valid_training_words.txt \
          icwb2-data/gold/{dataset_name[1:-1].lower()}_valid.utf8 "
                + f"exp/result/{exp_file}/valid.txt > score_valid.utf8"
            )

            f1_score = 0.0

            with open(f"./score_valid.utf8", "r") as f:
                a = f.readlines(0)
                for string in a[-5:]:
                    if F1.match(string) is not None:
                        if float("0." + F1.findall(string)[0]) > f1_score:
                            f1_score = float("0." + F1.findall(string)[0])
                        break
            with open(f"./exp/result/{exp_file}/valid_record", "a") as f:
                f.write(
                    f"Training step : {step} Dataset : {dataset_name[1:-1]} F1 score : {f1_score}\n"
                )
    return f1_score


def score(exp_file: str, criteria: str, step: int, dataset_token: str) -> float:
    """In this funciton, we first postprocess the testing data. The word that transformed \
        to 'a' or '0' should be recovered. Then we use the scoring scripts from sighan to \
        evaluate our model's performance.

    Args:
        exp_file (str): The file to store the evaluation result.
        criteria (str): Which dataset we are evaulating.
        step (int): The storing period of model.  
        dataset_token (str): The criterion token placed at the beginning of each datum.

    Returns:
        F1 score (flaot) : The F1 score result. 
        OOV recall (flaot) : The OOV recall result.
    """
    F1 = re.compile(r"=== F MEASURE:\s+0.(\d+)")
    OOV = re.compile(r"=== OOV Recall Rate:\s+0.(\d+)")

    if criteria == "[AS]":
        os.system(
            f"python3 -m MCCWS.script.postprocess \
        --original_test_gold_path ./icwb2-data/gold/as_test_gold.utf8 \
        --test_file ./exp/result/{exp_file}/{step}_{criteria}_{dataset_token}.txt"
        )
        os.system(
            "perl icwb2-data/scripts/score icwb2-data/gold/as_training_words.utf8 \
      icwb2-data/gold/as_test_gold.utf8 "
            + f"exp/result/{exp_file}/{step}_{criteria}_{dataset_token}.txt > score.utf8"
        )
    if criteria == "[CIT]":
        os.system(
            f"python3 -m MCCWS.script.postprocess \
        --original_test_gold_path ./icwb2-data/gold/cityu_test_gold.utf8 \
        --test_file ./exp/result/{exp_file}/{step}_{criteria}_{dataset_token}.txt"
        )
        os.system(
            "perl icwb2-data/scripts/score icwb2-data/gold/cityu_training_words.utf8 \
      icwb2-data/gold/cityu_test_gold.utf8 "
            + f"exp/result/{exp_file}/{step}_{criteria}_{dataset_token}.txt > score.utf8"
        )
    if criteria == "[MSR]":
        os.system(
            f"python3 -m MCCWS.script.postprocess \
        --original_test_gold_path ./icwb2-data/gold/msr_test_gold.utf8 \
        --test_file ./exp/result/{exp_file}/{step}_{criteria}_{dataset_token}.txt"
        )
        os.system(
            "perl icwb2-data/scripts/score icwb2-data/gold/msr_training_words.utf8 \
      icwb2-data/gold/msr_test_gold.utf8 "
            + f"exp/result/{exp_file}/{step}_{criteria}_{dataset_token}.txt > score.utf8"
        )
    if criteria == "[PKU]":
        print("1")
        os.system(
            f"python3 -m MCCWS.script.postprocess \
        --original_test_gold_path ./icwb2-data/gold/pku_test_gold.utf8 \
        --test_file ./exp/result/{exp_file}/{step}_{criteria}_{dataset_token}.txt"
        )
        os.system(
            "perl icwb2-data/scripts/score icwb2-data/gold/pku_training_words.utf8 \
      icwb2-data/gold/pku_test_gold.utf8 "
            + f"exp/result/{exp_file}/{step}_{criteria}_{dataset_token}.txt > score.utf8"
        )
    if criteria == "[CTB6]":
        os.system(
            f"python3 -m MCCWS.script.postprocess \
        --original_test_gold_path ./icwb2-data/gold/ctb6_test_gold.utf8 \
        --test_file ./exp/result/{exp_file}/{step}_{criteria}_{dataset_token}.txt"
        )
        os.system(
            "perl icwb2-data/scripts/score icwb2-data/gold/ctb6_training_words.utf8 \
      icwb2-data/gold/ctb6_test_gold.utf8 "
            + f"exp/result/{exp_file}/{step}_{criteria}_{dataset_token}.txt > score.utf8"
        )
    if criteria == "[CNC]":
        os.system(
            f"python3 -m MCCWS.script.postprocess \
        --original_test_gold_path ./icwb2-data/gold/cnc_test_gold.txt \
        --test_file ./exp/result/{exp_file}/{step}_{criteria}_{dataset_token}.txt"
        )
        os.system(
            "perl icwb2-data/scripts/score icwb2-data/gold/cnc_training_words.utf8 \
      icwb2-data/gold/cnc_test_gold.txt "
            + f"exp/result/{exp_file}/{step}_{criteria}_{dataset_token}.txt > score.utf8"
        )
    if criteria == "[SXU]":
        os.system(
            f"python3 -m MCCWS.script.postprocess \
        --original_test_gold_path ./icwb2-data/gold/sxu_test_gold.txt \
        --test_file ./exp/result/{exp_file}/{step}_{criteria}_{dataset_token}.txt"
        )
        os.system(
            "perl icwb2-data/scripts/score icwb2-data/gold/sxu_training_words.utf8 \
      icwb2-data/gold/sxu_test_gold.txt "
            + f"exp/result/{exp_file}/{step}_{criteria}_{dataset_token}.txt > score.utf8"
        )
    if criteria == "[UD]":
        os.system(
            f"python3 -m MCCWS.script.postprocess \
        --original_test_gold_path ./icwb2-data/gold/ud_test_gold.txt \
        --test_file ./exp/result/{exp_file}/{step}_{criteria}_{dataset_token}.txt"
        )
        os.system(
            "perl icwb2-data/scripts/score icwb2-data/gold/ud_training_words.utf8 \
      icwb2-data/gold/ud_test_gold.txt "
            + f"exp/result/{exp_file}/{step}_{criteria}_{dataset_token}.txt > score.utf8"
        )
    if criteria == "[WTB]":
        os.system(
            f"python3 -m MCCWS.script.postprocess \
        --original_test_gold_path ./icwb2-data/gold/wtb_test_gold.txt \
        --test_file ./exp/result/{exp_file}/{step}_{criteria}_{dataset_token}.txt"
        )
        os.system(
            "perl icwb2-data/scripts/score icwb2-data/gold/wtb_training_words.utf8 \
      icwb2-data/gold/wtb_test_gold.txt "
            + f"exp/result/{exp_file}/{step}_{criteria}_{dataset_token}.txt > score.utf8"
        )
    if criteria == "[ZX]":
        os.system(
            f"python3 -m MCCWS.script.postprocess \
        --original_test_gold_path ./icwb2-data/gold/zx_test_gold.txt \
        --test_file ./exp/result/{exp_file}/{step}_{criteria}_{dataset_token}.txt"
        )
        os.system(
            "perl icwb2-data/scripts/score icwb2-data/gold/zx_training_words.utf8 \
      icwb2-data/gold/zx_test_gold.txt "
            + f"exp/result/{exp_file}/{step}_{criteria}_{dataset_token}.txt > score.utf8"
        )
    with open(f"./score.utf8", "r") as f:
        a = f.readlines(0)
        for string in a[-5:]:
            if F1.match(string) is not None:
                F1_score = float("0." + F1.findall(string)[0])
            if OOV.match(string) is not None:
                OOV_recall = float("0." + OOV.findall(string)[0])
    return F1_score, OOV_recall

if __name__ == "__main__":
    # os.system(
    #     f"python -m MCCWS.script.postprocess \
    # --original_test_gold_path ../icwb2-data/gold/pku_test_gold.utf8 \
    # --test_file ../exp/result/cws_model/50000_[PKU]_[PKU].txt"
    # )

    # os.system(
    #     "perl ../icwb2-data/scripts/score ../icwb2-data/gold/pku_training_words.utf8 \
    #     ../icwb2-data/gold/pku_test_gold.utf8 "+ f"../exp/result/cws_model/50000_[PKU]_[PKU].txt > score.utf8"
    # )
 #    str="perl score ../gold/pku_training_words.utf8 \
 # ../gold/pku_test_gold.utf8 "+ f"../../exp/result/cws_model/50000_[PKU]_[PKU].txt > score.utf8"
    # print(str)

    strs="perl ../icwb2-data/scripts/score ../icwb2-data/gold/pku_training_words.utf8 \
        ../icwb2-data/gold/pku_test_gold.utf8 "+ f"../exp/result/cws_model/50000_[PKU]_[PKU].txt > score.utf8"
    print(strs)

    F1 = re.compile(r"=== F MEASURE:\s+0.(\d+)")
    OOV = re.compile(r"=== OOV Recall Rate:\s+0.(\d+)")
    F1_scores=[]
    OOV_recalls=[]
    with open(f"score.utf8", "r",encoding='utf-8') as f:
        a = f.readlines()
        print(a)
        for string in a[-5:]:
            if F1.match(string) is not None:
                F1_score = float("0." + F1.findall(string)[0])
                F1_scores.append(F1_score)
            if OOV.match(string) is not None:
                OOV_recall = float("0." + OOV.findall(string)[0])
                OOV_recalls.append(OOV_recall)

    print(F1_scores)
    print(OOV_recalls)
