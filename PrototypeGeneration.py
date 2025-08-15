import numpy as np
import io

"""
wrongclass:[>1,n,bool]
missPER:[0,n,0]
missMISC:[n,0,bool]
"""

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


import ast
import json


def read_list_from_file(file_path):
    """从txt文件读取列表格式字符串并转换为Python列表"""
    try:
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read().strip()

        # 处理常见的格式问题
        content = _clean_list_string(content)

        # 尝试使用JSON解析（更严格，适合标准JSON格式）
        try:
            parsed_list = json.loads(content)
            return parsed_list
        except json.JSONDecodeError:
            # 尝试使用ast.literal_eval（更灵活，适合Python风格的列表）
            try:
                parsed_list = ast.literal_eval(content)
                if isinstance(parsed_list, list):
                    return parsed_list
                else:
                    raise ValueError(f"解析结果不是列表: {type(parsed_list)}")
            except Exception as e:
                raise ValueError(f"无法解析内容: {content}. 错误: {str(e)}")

    except FileNotFoundError:
        raise FileNotFoundError(f"文件不存在: {file_path}")
    except Exception as e:
        raise Exception(f"处理文件时出错: {str(e)}")


def _clean_list_string(s):
    """清理列表字符串，处理常见的格式问题"""
    # 移除首尾可能存在的引号
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()

    # 处理可能包含的多余引号（例如：["a", "b"] 或 ['a', 'b']）
    # 注意：这里假设内容是一致的，要么全是单引号，要么全是双引号
    if s.startswith('[') and s.endswith(']'):
        # 检查是否有不平衡的引号
        if (s.count('"') % 2 != 0) or (s.count("'") % 2 != 0):
            # 简单处理：移除所有单引号（如果双引号数量是偶数）
            if s.count('"') % 2 == 0:
                s = s.replace("'", "")
            # 或移除所有双引号（如果单引号数量是偶数）
            elif s.count("'") % 2 == 0:
                s = s.replace('"', "")

    return s


def fliter_for_wrongclass(rating_list):
    rating_list_for_wrongclass = []

    for i in rating_list:
        element = []
        if i[0] != 0 and i[1] != 0 and i[0] <= 10 and i[1] <= 10:
            element.append(i[0])
            element.append(i[1])
            element.append(i[2])
            rating_list_for_wrongclass.append(element)

    return rating_list_for_wrongclass


def fliter_for_missTarget(rating_list):
    rating_list_for_miss = []
    for i in rating_list:
        element = []
        if i[0] == 0 and i[0] <= 10 and i[1] <= 10:
            element.append(i[0])
            element.append(i[1])
            element.append(i[2])
            rating_list_for_miss.append(element)

    return rating_list_for_miss


def fliter_for_missMISC(rating_list):
    rating_list_for_miss = []
    for i in rating_list:
        element = []
        if i[1] == 0 and i[0] <= 10 and i[1] <= 10:
            element.append(i[0])
            element.append(i[1])
            element.append(i[2])
            rating_list_for_miss.append(element)

    return rating_list_for_miss


def fliter_for_right(rating_list):
    rating_list_for_right = []
    for i in rating_list:
        if i[-1] == 1:
            element = []
            element.append(i[0])
            element.append(i[1])
            rating_list_for_right.append(element)

    return rating_list_for_right


def fliter_for_wrong(rating_list):
    rating_list_for_wrong = []
    for i in rating_list:
        if i[-1] == 0:
            element = []
            element.append(i[0])
            element.append(i[1])
            rating_list_for_wrong.append(element)

    return rating_list_for_wrong




def calculate_prototype(rating_list):


    data = np.array(rating_list)


    covariance_matrix = np.cov(data, rowvar=False)


    mean_point = np.mean(rating_list, 0)


    distances = np.linalg.norm(data - mean_point, axis=1)

    mean_distance = np.mean(distances)

    dispersion = sigmoid(np.trace(covariance_matrix))

    cov = np.cov(data.T)

    return np.mean(rating_list, 0), dispersion, mean_distance, cov


def normList(raw_list):
    normls = []
    for i in raw_list:
        if len(i) == 1 and 0 <= i[0][0] <= 10 and 0 <= i[0][1] <= 10:
            normls.append(i[0])
        elif len(i) > 1:
            temp_list_right = []
            temp_list_wrong = []
            # for j in i:
            #     if j[0]<=10 and j[1]<=10:
            #         if j[-1] == 1:
            #             temp_list_right.append(j)
            #         else:
            #             temp_list_wrong.append(j)
            # if len(temp_list_right)>1:
            #     normls.append(np.mean(temp_list_right, 0).tolist())
            # if len(temp_list_wrong)>1:
            #     normls.append(np.mean(temp_list_wrong, 0).tolist())

            for j in i:
                if 0 <= j[0] <= 10 and 0 <= j[1] <= 10:
                    normls.append(j)

    return normls


def Proto_Gen_missMISC(llm_name, type):
    file_path = ''
    raw_list = read_list_from_file(file_path)
    normls = normList(raw_list)

    # for DT & ST
    # missMISC
    rating_list_for_missMISC = fliter_for_missMISC(normls)
    rating_list_for_missMISC_right = fliter_for_right(rating_list_for_missMISC)
    rating_list_for_missMISC_wrong = fliter_for_wrong(rating_list_for_missMISC)

    prototype_for_missMISC_right, prototype_for_missMISC_right_disp, prototype_for_missMISC_right_meandist, prototype_for_missMISC_right_cov = calculate_prototype(
        rating_list_for_missMISC_right)
    prototype_for_missMISC_wrong, prototype_for_missMISC_wrong_disp, prototype_for_missMISC_wrong_meandist, prototype_for_missMISC_wrong_cov = calculate_prototype(
        rating_list_for_missMISC_wrong)

    return prototype_for_missMISC_right, prototype_for_missMISC_right_disp, prototype_for_missMISC_right_meandist, prototype_for_missMISC_right_cov, prototype_for_missMISC_wrong, prototype_for_missMISC_wrong_disp, prototype_for_missMISC_wrong_meandist, prototype_for_missMISC_wrong_cov


def Proto_Gen_missTarget(llm_name, type):
    file_path = ''
    # file_path = fr'C:\ProtoFilter\Rating\Location\phi4.txt'
    raw_list = read_list_from_file(file_path)
    normls = normList(raw_list)

    # missTarget
    rating_list_for_missTarget = fliter_for_missTarget(normls)
    rating_list_for_missTarget_right = fliter_for_right(rating_list_for_missTarget)
    rating_list_for_missTarget_wrong = fliter_for_wrong(rating_list_for_missTarget)

    prototype_for_missTarget_right, prototype_for_missTarget_right_disp, prototype_for_missTarget_right_meandist, prototype_for_missTarget_right_cov = calculate_prototype(
        rating_list_for_missTarget_right)
    prototype_for_missTarget_wrong, prototype_for_missTarget_wrong_disp, prototype_for_missTarget_wrong_meandist, prototype_for_missTarget_wrong_cov = calculate_prototype(
        rating_list_for_missTarget_wrong)

    return prototype_for_missTarget_right, prototype_for_missTarget_right_disp, prototype_for_missTarget_right_meandist, prototype_for_missTarget_right_cov, prototype_for_missTarget_wrong, prototype_for_missTarget_wrong_disp, prototype_for_missTarget_wrong_meandist, prototype_for_missTarget_wrong_cov


def Proto_Gen_wrongclass(llm_name, type):
    file_path = ''
    raw_list = read_list_from_file(file_path)
    normls = normList(raw_list)

    rating_list_for_wrongclass = fliter_for_wrongclass(normls)

    rating_list_for_wrongclass_right = fliter_for_right(rating_list_for_wrongclass)
    rating_list_for_wrongclass_wrong = fliter_for_wrong(rating_list_for_wrongclass)


    prototype_for_wrongclass_right, prototype_for_wrongclass_right_disp, prototype_for_wrongclass_right_meandist, prototype_for_wrongclass_right_cov = calculate_prototype(
        rating_list_for_wrongclass_right)
    prototype_for_wrongclass_wrong, prototype_for_wrongclass_wrong_disp, prototype_for_wrongclass_wrong_meandist, prototype_for_wrongclass_wrong_cov = calculate_prototype(
        rating_list_for_wrongclass_wrong)


    return prototype_for_wrongclass_right, prototype_for_wrongclass_right_disp, prototype_for_wrongclass_right_meandist, prototype_for_wrongclass_right_cov,prototype_for_wrongclass_wrong, prototype_for_wrongclass_wrong_disp, prototype_for_wrongclass_wrong_meandist, prototype_for_wrongclass_wrong_cov


























