from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseOutputParser
import pandas as pd
import openpyxl
from nerEva import NEREval


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


def remove_think(s):
    if "</think>" in s:
        return s.split("</think>")[-1]
    else:
        return s

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


def NER(llm, type, sheet, LD):
    
    label = type[:3].upper()
    
    line_num = 1

    # num. of correctly recogized entities
    llm_entity_right_num = 0

    # num. of all recogized entities
    llm_entity_num = 0

    # num. of type misidentification entities
    wrong_class= 0

    # num. of missed entities
    miss= 0

    # num. of incorrectly recognized left boundary of entities
    inside_wrong = 0

    # num. of incorrectly recognized right boundary of entities
    head_wrong = 0

    # num. of ground truth entities
    entity_num = 0

    F1 = 0

    for row in sheet.iter_rows(min_row=2, values_only=True):

        print(f"line:{line_num}")

        # Sentence 
        cell_sentence = row[0]
        # True Entity 
        if pd.isna(row[1]):
            cell_entity = []
        elif ', ' in row[1]:
            cell_entity = row[1].split(', ')
        else:
            cell_entity = str(row[1]).replace(".", "")
        # True Entity type
        if pd.isna(row[2]):
            cell_class = []
        elif ', ' in row[2]:
            cell_class = row[2].split(', ')
        else:
            cell_class = str(row[2])

        true_entity_list = extract_entities(cell_entity, cell_class, label)
        true_entity_list = list(set([s.lower() for s in true_entity_list]))
        entity_num_sent = len(true_entity_list)


        template = """The following sentence may exist entities of {type} type.
        -If there are entities, please extract entities of {type} type and only respond the extracted entities in the format: "Entity" for only one entity or "Entity, Entity" for two or more entities, without any other words.
        -If there are no extracted entities, please respond with an empty string: "" without any other words.
        sentence：{sentence}"""

        # template = """The following sentence may exist entities of {type} type. Here is the information of entity type: {type}: {label_descri}
        #         -If there are entities, please extract entities of {type} type and only respond the extracted entities in the format: "Entity" for only one entity or "Entity, Entity" for two or more entities, without any other words.
        #         -If there are no extracted entities, please respond with an empty string: "" without any other words.
        #         sentence：{sentence}"""

        prompt = ChatPromptTemplate.from_template(template)
        output_parser = CommaSeparatedListOutputParser()
        chain = prompt | llm | output_parser
        res = chain.invoke({"type": type,"sentence": cell_sentence}).replace('"', '').replace("'", "").replace("\n","").replace("* ","").replace("*", "").replace(".", "")
        res = remove_before_last_colon(res)
        res = remove_think(res)

        if 'This ' not in res and 'There ' not in res and res != '' and ' no ' not in res:
                
            predict_entity_list = [item.strip() for item in res.split(',')]
            predict_entity_list = [item for item in predict_entity_list if item not in ('', []) and item is not None]
            predict_entity_list = list(set([s.lower() for s in predict_entity_list]))
            print(f"predict_entity_list: {predict_entity_list}")
            llm_entity_num_sent = len(predict_entity_list)


            print(f"true_entity_list: {true_entity_list}")

            head_wrong_sent, inside_wrong_sent, wrong_class_sent, miss_sent = NEREval(predict_entity_list,true_entity_list)

            llm_entity_right_num_sent = entity_num_sent -  head_wrong_sent - inside_wrong_sent - miss_sent

        else:

            llm_entity_num_sent = 0
            head_wrong_sent = 0
            inside_wrong_sent = 0
            wrong_class_sent = 0
            miss_sent = entity_num_sent
            llm_entity_right_num_sent = 0

        llm_entity_num += llm_entity_num_sent
        head_wrong += head_wrong_sent
        inside_wrong += inside_wrong_sent
        wrong_class += wrong_class_sent
        miss += miss_sent
        llm_entity_right_num += llm_entity_right_num_sent
        entity_num += entity_num_sent

        if llm_entity_num>0 and entity_num>0:
                P = llm_entity_right_num/llm_entity_num
                R = llm_entity_right_num/entity_num
                if P+R > 0:
                    F1 = (2*P*R)/(P+R)

        print(f"entity_num: {entity_num}")
        print(f"llm_entity_right_num: {llm_entity_right_num}")
        print(f"llm_entity_num: {llm_entity_num}")
        print(f"wrong_class_by_lmm: {wrong_class}")
        print(f"inside_wrong_by_lmm: {inside_wrong}")
        print(f"head_wrong_by_lmm: {head_wrong}")
        print(f"miss_by_lmm: {miss}")
        print(f"F1: {F1}")
        print("---" * 30)

        line_num += 1

    result = f"entity_num: {entity_num}\nllm_entity_right_num: {llm_entity_right_num}\nllm_entity_num: {llm_entity_num}\nwrong_class_by_lmm: {wrong_class}\ninside_wrong_by_lmm: {inside_wrong}\nhead_wrong_by_lmm: {head_wrong}\nmiss_by_lmm: {miss}\nF1: {F1}"

    return result


if __name__ == "__main__":

    dataset_list = ["conll03_test","conll03_valid","wnut17_test"]
    llm_list = ["llama3.1", "gemma2", "phi4", "qwen2.5"]
    type_list = ["Person", "Organizaiton", "Location"]
    LD_list = [
        "This category includes names of persons, such as individual people or groups of people with personal names.",
        "This category includes names of formally structured groups, such as companies, institutions, agencies, or teams, within text.",
        "This category includes names of specific geographical places, such as cities, countries, regions, or landmarks, within text."]

    for dataset in dataset_list:
        workbook = openpyxl.load_workbook('../dataset/'+dataset+'.xlsx')
        sheet = workbook.active

        for llm_name in llm_list:
            llm = ChatOllama(model=llm_name)
            for i in range(len(type_list)):
                result = NER(llm, type_list[i], sheet, LD_list[i])
                with open(f"../Result_v/{dataset}/{type_list[i]}/{llm_name}.txt", "w", encoding="utf-8") as f:
                    f.write(result)  # 直接将列表转为字符串写入






























