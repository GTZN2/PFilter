# -*- coding: utf-8 -*-
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseOutputParser
import numpy as np
import pandas as pd
import math
import openpyxl
import re
from PrototypeGeneration import Proto_Gen_missMISC, Proto_Gen_missTarget
import numpy as np
from numpy.linalg import det, inv
from nerEva import NEREval
import requests
import json

class CommaSeparatedListOutputParser(BaseOutputParser):
    """Parse the output of an LLM call to a comma-separated list."""

    def parse(self, text: str):
        """Parse the output of an LLM call."""

        return text.strip()


def mahalanobis_distance(x, mu, sigma_inv):

    delta = x - mu
    return np.sqrt(delta.T @ sigma_inv @ delta)


def gaussian_probability(x, mean, cov):
    """
    计算点x在 multivariate 高斯分布中的概率密度

    参数:
    x -- 待计算的点
    mean -- 分布的均值向量
    cov -- 分布的协方差矩阵

    返回:
    概率密度值
    """
    n = x.shape[0]  # 维度

    # 计算协方差矩阵的行列式
    cov_det = det(cov)
    if cov_det == 0:
        # 处理奇异矩阵（添加微小扰动）
        cov = cov + np.eye(n) * 1e-6
        cov_det = det(cov)

    # 计算协方差矩阵的逆
    cov_inv = inv(cov)

    # 计算指数部分
    x_minus_mean = x - mean
    exponent = -0.5 * np.dot(np.dot(x_minus_mean.T, cov_inv), x_minus_mean)

    # 计算概率密度
    denominator = np.sqrt((2 * np.pi) ** n * cov_det)
    probability = (1 / denominator) * np.exp(exponent)

    return probability


def Filter_for_missMISC(rating, prototype_for_missMISC_right, prototype_for_missMISC_right_disp, prototype_for_missMISC_right_meandist, prototype_for_missMISC_right_cov, prototype_for_missMISC_wrong, prototype_for_missMISC_wrong_disp, prototype_for_missMISC_wrong_meandist, prototype_for_missMISC_wrong_cov):


    new_point = np.array(rating)

    # 计算新点在两个分布中的概率密度
    prob_right = gaussian_probability(new_point, prototype_for_missMISC_right, prototype_for_missMISC_right_cov)
    prob_wrong = gaussian_probability(new_point, prototype_for_missMISC_wrong, prototype_for_missMISC_wrong_cov)

    if rating[0] <= round(prototype_for_missMISC_wrong[0]) or (rating[0] > round(prototype_for_missMISC_wrong[0]) and rating[0] <= (prototype_for_missMISC_right[0]-prototype_for_missMISC_right_meandist) and prob_right < prob_wrong):

        return True
    else:
        return False

def Filter_for_missTarget(rating, prototype_for_missTarget_right, prototype_for_missTarget_right_disp, prototype_for_missTarget_right_meandist, prototype_for_missTarget_right_cov, prototype_for_missTarget_wrong, prototype_for_missTarget_wrong_disp, prototype_for_missTarget_wrong_meandist, prototype_for_missTarget_wrong_cov):

    new_point = np.array(rating)

    # 计算新点在两个分布中的概率密度
    prob_right = gaussian_probability(new_point, prototype_for_missTarget_right, prototype_for_missTarget_right_cov)
    prob_wrong = gaussian_probability(new_point, prototype_for_missTarget_wrong, prototype_for_missTarget_wrong_cov)

    if rating[1] <= round(prototype_for_missTarget_wrong[1]) or (rating[1] > round(prototype_for_missTarget_wrong[1]) and rating[1] <= round(prototype_for_missTarget_right[1]-prototype_for_missTarget_right_meandist) and prob_right < prob_wrong):

        return True
    else:
        return False

def remove_subsets(strings):

    to_remove = []
    for i, s1 in enumerate(strings):
        for s2 in strings:
            if s1 != s2 and s1 in s2:
                to_remove.append(s1)
                break


    to_remove = set(to_remove)


    result = [s for s in strings if s not in to_remove]

    return result

def extract_entities(listA, listB, tag):
        entities = []
        current_entity = []

        for entity, label in zip(listA, listB):
            if label == f'B-{tag}':
                if current_entity:
                    entities.append(' '.join(current_entity))
                current_entity = [entity]
            elif label == f'I-{tag}':
                current_entity.append(entity)

        if current_entity:
            entities.append(' '.join(current_entity))

        return entities

def remove_before_last_colon(s):
    index = s.rfind(':')
    if index != -1:
        return s[index + 1:]
    return s

def process_entity_string(s):
    # 步骤1：检查是否仅有两对"[]"并提取内容
    if s.count("[") != 2 or s.count("]") != 2:
        return ""

    # 提取两个中括号内的内容
    list_contents = re.findall(r'\[(.*?)\]', s)
    if len(list_contents) != 2:
        return ""

    listA_str, listB_str = list_contents
    listA = [item.strip() for item in listA_str.split(',') if item.strip()]
    listB = [item.strip() for item in listB_str.split(',') if item.strip()]

    # 步骤2：处理listA得到list1和list2
    list1, list2 = [], []
    for item in listA:
        if '//' in item:
            parts = item.split('//', 1)  # 只分割一次
            content, rating_str = parts[0].strip(), parts[1].strip()
            if rating_str.isdigit():
                list1.append(content)
                list2.append(int(rating_str))

    # 步骤3：处理listB得到list3和list4
    list3, list4 = [], []
    for item in listB:
        if '//' in item:
            parts = item.split('//', 1)
            content, rating_str = parts[0].strip(), parts[1].strip()
            if rating_str.isdigit():
                list3.append(content)
                list4.append(int(rating_str))

    return list1, list2, list3, list4

def filter_non_int_convertible_elements(lst):
    indices_to_remove = []
    for index, element in enumerate(lst):
        try:
            float(element)
        except ValueError:
            indices_to_remove.append(index)

    for index in reversed(indices_to_remove):
        del lst[index]

    return lst, indices_to_remove

def NER_PF(type, sheet, LD):
    label = type[:3].upper()

    Baseurl = ""
    Skey = ""
    url = Baseurl + "/v1/chat/completions"
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {Skey}',
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
        'Content-Type': 'application/json'
    }


    line_num = 1

    # num. of correctly recogized entities
    llm_entity_right_num = 0

    # num. of all recogized entities
    llm_entity_num = 0

    # num. of type misidentification entities
    wrong_class = 0

    # num. of missed entities
    miss = 0

    # num. of incorrectly recognized left boundary of entities
    inside_wrong = 0

    # num. of incorrectly recognized right boundary of entities
    head_wrong = 0

    # num. of ground truth entities
    entity_num = 0

    F1 = 0

    prototype_for_missMISC_right, prototype_for_missMISC_right_disp, prototype_for_missMISC_right_meandist, prototype_for_missMISC_right_cov, prototype_for_missMISC_wrong, prototype_for_missMISC_wrong_disp, prototype_for_missMISC_wrong_meandist, prototype_for_missMISC_wrong_cov = Proto_Gen_missMISC(
        "GPT", type)
    prototype_for_missTarget_right, prototype_for_missTarget_right_disp, prototype_for_missTarget_right_meandist, prototype_for_missTarget_right_cov, prototype_for_missTarget_wrong, prototype_for_missTarget_wrong_disp, prototype_for_missTarget_wrong_meandist, prototype_for_missTarget_wrong_cov = Proto_Gen_missTarget("GPT", type)


    for row in sheet.iter_rows(min_row=2, values_only=True):

        print("line:" + str(line_num))

        cell_sentence = row[0]
        if pd.isna(row[1]):
            cell_entity = []
        elif ', ' in row[1]:
            cell_entity = row[1].split(', ')
        else:
            cell_entity = str(row[1]).replace(".", "")

        if pd.isna(row[2]):
            cell_class = []
        elif ', ' in row[2]:
            cell_class = row[2].split(', ')
        else:
            cell_class = str(row[2])

        true_entity_list = extract_entities(cell_entity, cell_class, label)
        true_entity_list = list(set([s.lower() for s in true_entity_list]))
        entity_num_sent = len(true_entity_list)

        template0 = """The following sentence may exist entities of {type} type.
        -If there are entities, please extract entities of {type} type and only respond the extracted entities in the format: "Entity" for only one entity or "Entity, Entity" for two or more entities, without any other words.
        -If there are no extracted entities, please respond with an empty string: "" without any other words.
        sentence：{sentence}
                """

        payload = json.dumps({
            "model": "gpt-4",
            "messages": [
                {
                    "role": "user",
                    "content": template0.format(sentence=cell_sentence, type=type)
                }
            ]
        })
        response = requests.request("POST", url, headers=headers, data=payload).json()

        A = response['choices'][0]['message'][
            'content']

        res0 = A.replace('"', '').replace("'", "").replace("\n", "").replace("* ", "").replace("*", "").replace(
            ".", "")

        res0 = remove_before_last_colon(res0)

        if 'This ' not in res0 and 'There ' not in res0 and res0 != '' and ' no ' not in res0:
            v_predict_entity_list = [item.strip() for item in res0.split(',')]
            v_predict_entity_list = [item for item in v_predict_entity_list if item not in ('', []) and item is not None]
            v_predict_entity_list = list(set([s.lower() for s in v_predict_entity_list]))

            print(f"v_predict_entity_list: {v_predict_entity_list}")

            template1 = """ 
                                Here is the information of entity type {type}: {label_descri}
                                Please complete two tasks and output two lists respectively: 
                                1. For each entity in the following list, rate its relevance to the **type {type}** on a scale of 1 to 10. Each element in the list should be in the format "Entity//Rating". If the Entity_list is empty, this list is empty []. 
                                2. From the following sentence, extract all entities **other than the {type} type**, and rate their relevance to the **type {type}** on a scale of 1 to 10. Each element in the list should be in the format "Entity//Rating". If there are no such entities, this list is empty []. 
                                **Only output the two lists, with no other words.**

                                Output Format:
                                [Entity1//Rating1, Entity2//Rating2,...], [Entity3//Rating3, Entity4//Rating4,...]

                                list: {Entity_list}      
                                sentence：{sentence}
                                """
            predict_entity_list = []
            predict_rating_list = []
            predict_MISC_entity_list = []
            predict_MISC_rating_list = []

            payload = json.dumps({
                "model": "gpt-3.5-turbo",
                "messages": [
                    {
                        "role": "user",
                        "content": template1.format(sentence=cell_sentence, type=type, label_descri=LD, Entity_list = str(v_predict_entity_list))
                    }
                ]
            })
            response = requests.request("POST", url, headers=headers, data=payload).json()['choices'][0]['message'][
                'content']

            res1 = response.replace('"', '').replace("'", "").replace("\n", "").replace("* ", "").replace("*",
                                                                                                         "").replace(
                ".", "")

            if process_entity_string(res1) != "":
                predict_entity_list, predict_rating_list, predict_MISC_entity_list, predict_MISC_rating_list = process_entity_string(
                    res1)

            predict_entity_list = [s.lower() for s in predict_entity_list]
            predict_MISC_entity_list = [s.lower() for s in predict_MISC_entity_list]

            final_prediction_list = v_predict_entity_list.copy()
            if len(predict_entity_list) > 0:
                for i in range(len(predict_entity_list)):
                    if predict_entity_list[i] not in predict_MISC_entity_list:
                        if Filter_for_missMISC([float(predict_rating_list[i]), 0], prototype_for_missMISC_right,
                                               prototype_for_missMISC_right_disp, prototype_for_missMISC_right_meandist,
                                               prototype_for_missMISC_right_cov, prototype_for_missMISC_wrong,
                                               prototype_for_missMISC_wrong_disp, prototype_for_missMISC_wrong_meandist,
                                               prototype_for_missMISC_wrong_cov):
                            if predict_entity_list[i] in final_prediction_list:
                                final_prediction_list.remove(predict_entity_list[i])

            if len(predict_MISC_entity_list) > 0:
                for j in range(len(predict_MISC_entity_list)):
                    if predict_MISC_entity_list[j] not in predict_entity_list:
                        if Filter_for_missTarget([0, float(predict_MISC_rating_list[j])],
                                                 prototype_for_missTarget_right, prototype_for_missTarget_right_disp,
                                                 prototype_for_missTarget_right_meandist,
                                                 prototype_for_missTarget_right_cov, prototype_for_missTarget_wrong,
                                                 prototype_for_missTarget_wrong_disp,
                                                 prototype_for_missTarget_wrong_meandist,
                                                 prototype_for_missTarget_wrong_cov):
                            if predict_MISC_entity_list[j] in final_prediction_list:
                                final_prediction_list.remove(predict_MISC_entity_list[j])

            final_prediction_list = list(set([s.lower() for s in final_prediction_list]))
            print(f"final_prediction_list: {final_prediction_list}")
            llm_entity_num_sent = len(final_prediction_list)


            print(f"true_entity_list: {true_entity_list}")

            head_wrong_sent, inside_wrong_sent, wrong_class_sent, miss_sent = NEREval(final_prediction_list,
                                                                                      true_entity_list)

            llm_entity_num += llm_entity_num_sent
            head_wrong += head_wrong_sent
            inside_wrong += inside_wrong_sent
            wrong_class += wrong_class_sent
            miss += miss_sent
            llm_entity_right_num += entity_num_sent - head_wrong_sent - inside_wrong_sent - miss_sent
            entity_num += entity_num_sent

        else:
            llm_entity_num += 0
            head_wrong += 0
            inside_wrong += 0
            wrong_class += 0
            miss += entity_num_sent
            llm_entity_right_num += 0
            entity_num += entity_num_sent



        if llm_entity_num > 0 and entity_num > 0:
            P = llm_entity_right_num / llm_entity_num
            R = llm_entity_right_num / entity_num
            if P + R > 0:
                F1 = (2 * P * R) / (P + R)

        print(f"entity_num: {entity_num}\nllm_entity_right_num: {llm_entity_right_num}\nllm_entity_num: {llm_entity_num}\nwrong_class_by_lmm: {wrong_class}\ninside_wrong_by_lmm: {inside_wrong}\nhead_wrong_by_lmm: {head_wrong}\nmiss_by_lmm: {miss}\nF1: {F1}")
        print("---" * 30)

        line_num += 1

    result = f"entity_num: {entity_num}\nllm_entity_right_num: {llm_entity_right_num}\nllm_entity_num: {llm_entity_num}\nwrong_class_by_lmm: {wrong_class}\ninside_wrong_by_lmm: {inside_wrong}\nhead_wrong_by_lmm: {head_wrong}\nmiss_by_lmm: {miss}\nF1: {F1}"

    return result

if __name__ == "__main__":

    dataset_list = ["conll03_test","conll03_valid","wnut17_test"]
    type_list = ["Person", "Organizaiton", "Location"]
    LD_list = [
        "This category includes names of persons, such as individual people or groups of people with personal names.",
        "This category includes names of formally structured groups, such as companies, institutions, agencies, or teams, within text.",
        "This category includes names of specific geographical places, such as cities, countries, regions, or landmarks, within text."]

    for dataset in dataset_list:
        workbook = openpyxl.load_workbook('../dataset/' + dataset + '.xlsx')
        sheet = workbook.active


        for i in range(len(type_list)):
                result = NER_PF(type_list[i], sheet, LD_list[i])
                with open(f"../Result_pf/{dataset}/{type_list[i]}/GPT3.5.txt", "w", encoding="utf-8") as f:
                    f.write(result)


















