# -*- coding: utf-8 -*-

from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseOutputParser
import pandas as pd
import openpyxl
import re


class CommaSeparatedListOutputParser(BaseOutputParser):
    """Parse the output of an LLM call to a comma-separated list."""

    def parse(self, text: str):
        """Parse the output of an LLM call."""

        return text.strip()


def remove_before_last_colon(s):
    index = s.rfind(':')
    if index != -1:
        return s[index + 1:]
    return s


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


def find_matching_entity(entity, true_list):
    """检查实体是否与true_list中的任何实体有word重叠"""
    # 将实体按空格分割成单词（如果实体是短语）
    entity_words = set(entity.split())

    for true_entity in true_list:
        true_words = set(true_entity.split())
        # 如果有任何单词重叠
        if entity_words.intersection(true_words):
            return True
    return False


def process_entity_lists(target_list, cross_list, target_rating_list, cross_rating_list, true_list):
    """处理实体列表并生成结果列表"""
    result = []
    processed_entities = set()  # 记录已处理的实体，避免重复

    # 第一部分：遍历target_list
    for t, entity in enumerate(target_list):
        try:
            # 情况1：cross_list中存在相同实体
            c = cross_list.index(entity)
            St = target_rating_list[t]
            Sc = cross_rating_list[c]
        except ValueError:
            # 情况2：cross_list中不存在相同实体
            St = target_rating_list[t]
            Sc = 0

        # 判断实体是否在true_list中有匹配
        o = 1 if find_matching_entity(entity, true_list) else 0

        # 添加到结果列表
        result.append([St, Sc, o])
        processed_entities.add(entity)

    # 第二部分：遍历cross_list，处理target_list中不存在的实体
    for c, entity in enumerate(cross_list):
        if entity not in processed_entities:
            Sc = cross_rating_list[c]
            St = 0
            o = 1 if find_matching_entity(entity, true_list) else 0
            result.append([St, Sc, o])

    return result


def process_entity_string(s):
    # 步骤1：检查是否仅有两对"[]"并提取内容
    if s.count("[") != 1 or s.count("]") != 1:
        return ""

    # 提取两个中括号内的内容
    list_contents = re.findall(r'\[(.*?)\]', s)
    if len(list_contents) != 1:
        return ""

    list_str = list_contents
    list = [item.strip() for item in list_str.split(',') if item.strip()]

    # 步骤2：处理listA得到list1和list2
    list1, list2 = [], []
    for item in list:
        if '//' in item:
            parts = item.split('//', 1)  # 只分割一次
            content, rating_str = parts[0].strip(), parts[1].strip()
            if rating_str.isdigit():
                list1.append(content)
                list2.append(int(rating_str))

    return list1, list2


def RatingCollect(llm, type, sheet, LD):
    label = type[:3].upper()

    all_rating_list = []

    line_num = 1

    for row in sheet.iter_rows(min_row=2, values_only=True):

        print("line:" + str(line_num))

        cell_sentence = row[0]
        if pd.isna(row[1]):
            cell_entity = []
        elif ', ' in row[1]:
            cell_entity = row[1].split(', ')
        else:
            cell_entity = str(row[1])

        if pd.isna(row[2]):
            cell_class = []
        elif ', ' in row[2]:
            cell_class = row[2].split(', ')
        else:
            cell_class = str(row[2])

        template1 = """
                                  The following sentence may exist entities of the type {type}. Here is the entity type information: {type}: {label_descri}

                                  If there are entities:
                                  -Please extract all entities of the type {type} and rate the relevance of extracted entities to the type {type} on a scale of 1 to 10.
                                  -Please only respond all extracted entities with rating strictly in the format: "Entity//Rating" for only one phrase or "Entity//Rating, Entity//Rating" for two or more phrases, without any other words.

                                  sentence：{sentence}

                                  """

        predict_entity_list = []
        predict_rating_list = []


        prompt1 = ChatPromptTemplate.from_template(template1)
        output_parser = CommaSeparatedListOutputParser()
        chain1 = prompt1 | llm | output_parser
        res1 = chain1.invoke({"label_descri": LD, "type": type, "sentence": cell_sentence}).replace('"', '').replace("'",
                                                                                                                   "").replace(
            "\n", "").replace("* ", "").replace("*", "").replace(".", "")

        print(res1)

        if process_entity_string(res1) != "":
            predict_entity_list, predict_rating_list = process_entity_string(
                res1)



        template2 = """
                                  The following sentence may contain entities other than those of the {type} type. Here is the entity type information: {type}: {label_descri}

                                  If there are entities:
                                  -Please extract all entities other than those of the {type} type and rate the relevance of extracted entities to the type {type} on a scale of 1 to 10.
                                  -Please only respond all extracted entities with rating strictly in the format: "Entity//Rating" for only one phrase or "Entity//Rating, Entity//Rating" for two or more phrases, without any other words.

                                  sentence：{sentence}

                                  """

        predict_MISC_entity_list = []
        predict_MISC_rating_list = []

        prompt2 = ChatPromptTemplate.from_template(template2)
        output_parser = CommaSeparatedListOutputParser()
        chain2 = prompt2 | llm | output_parser
        res2 = chain2.invoke({"label_descri": LD, "type": type, "sentence": cell_sentence}).replace('"', '').replace("'",
                                                                                                                   "").replace(
            "\n", "").replace("* ", "").replace("*", "").replace(".", "")

        print(res2)

        if process_entity_string(res2) != "":
            predict_MISC_entity_list, predict_MISC_rating_list = process_entity_string(
                res2)

        predict_entity_list = [s.lower() for s in predict_entity_list]
        predict_MISC_entity_list = [s.lower() for s in predict_MISC_entity_list]

        true_entity_list = extract_entities(cell_entity, cell_class, label)
        true_entity_list = list(set([s.lower() for s in true_entity_list]))

        rating_sent = process_entity_lists(predict_entity_list, predict_MISC_entity_list, predict_rating_list,
                                           predict_MISC_rating_list, true_entity_list)

        all_rating_list.append(rating_sent)
        print(f"all_rating_list: {all_rating_list}")

        line_num += 1

    return all_rating_list


if __name__ == "__main__":

    dataset_list = ["conll03_test"]
    llm_list = ["gemma2", "phi4", "llama3.1", "qwen2.5"]
    type_list = ["Person", "Organizaiton", "Location"]
    LD_list = [
        "This category includes names of persons, such as individual people or groups of people with personal names.",
        "This category includes names of formally structured groups, such as companies, institutions, agencies, or teams, within text.",
        "This category includes names of specific geographical places, such as cities, countries, regions, or landmarks, within text."]

    for dataset in dataset_list:
        workbook = openpyxl.load_workbook('../dataset/' + dataset + '.xlsx')
        sheet = workbook.active

        for llm_name in llm_list:
            llm = ChatOllama(model=llm_name)
            for i in range(len(type_list)):
                all_rating_list = RatingCollect(llm, type_list[i], sheet, LD_list[i])
                with open(rf"C:\ProtoFilter\Rating\{type_list[i]}\{llm_name}.txt", "w", encoding="utf-8") as f:
                    f.write(str(all_rating_list))  # 直接将列表转为字符串写入





















