def ner_error_analysis(pred_list, true_list):
    """
    分析NER预测结果的各类错误：左边界错误、右边界错误、漏检实体、预测无重叠实体

    参数:
    pred_list (list): 预测实体列表，每个实体为字符串（如["New York", "Los Angeles"]）
    true_list (list): 真实实体列表，每个实体为字符串（如["New York City", "Houston"]）

    返回:
    dict: 包含各类错误的数量及具体实体
        left_error_count: 左边界错误数量
        right_error_count: 右边界错误数量
        miss_count: 漏检实体数量（真实存在但未被预测）
        pred_no_overlap_count: 预测无重叠实体数量（预测但与真实无任何单词重叠）
        details: 各类错误的具体实体明细
    """
    # 预处理：将实体拆分为单词列表（方便比较）
    pred_entities = [entity.split() for entity in pred_list]  # 如["New York"] → [["New", "York"]]
    true_entities = [entity.split() for entity in true_list]
    pred_len = len(pred_entities)
    true_len = len(true_entities)

    # 记录已匹配的索引（避免重复匹配）
    matched_pred = set()  # 已匹配的预测实体索引
    matched_true = set()  # 已匹配的真实实体索引
    matches = []  # 匹配对：(pred_idx, true_idx)

    # 步骤1：先匹配完全相同的实体（所有单词一致）
    for p_idx in range(pred_len):
        if p_idx in matched_pred:
            continue
        pred_words = pred_entities[p_idx]
        for t_idx in range(true_len):
            if t_idx in matched_true and true_entities[t_idx] != pred_words:
                continue
            if true_entities[t_idx] == pred_words:
                matches.append((p_idx, t_idx))
                matched_pred.add(p_idx)
                matched_true.add(t_idx)
                break

    # 步骤2：对剩余实体，按单词重叠度（重叠单词数量）匹配
    # 计算两个实体的单词重叠数
    def overlap_count(pred_words, true_words):
        return len(set(pred_words) & set(true_words))

    # 处理未匹配的预测实体
    for p_idx in range(pred_len):
        if p_idx in matched_pred:
            continue
        pred_words = pred_entities[p_idx]
        best_overlap = 0
        best_t_idx = -1
        # 找重叠最多的真实实体
        for t_idx in range(true_len):
            if t_idx in matched_true:
                continue
            current_overlap = overlap_count(pred_words, true_entities[t_idx])
            if current_overlap > best_overlap:
                best_overlap = current_overlap
                best_t_idx = t_idx
        if best_overlap > 0:  # 有重叠才匹配
            matches.append((p_idx, best_t_idx))
            matched_pred.add(p_idx)
            matched_true.add(best_t_idx)

    # 统计边界错误（基于匹配对）
    left_errors = []  # 左边界错误的实体对 (pred, true)
    right_errors = []  # 右边界错误的实体对 (pred, true)
    for p_idx, t_idx in matches:
        pred_words = pred_entities[p_idx]
        true_words = true_entities[t_idx]
        pred_entity = pred_list[p_idx]
        true_entity = true_list[t_idx]

        # 左边界错误：第一个单词不同
        if pred_words[0] != true_words[0]:
            left_errors.append((pred_entity, true_entity))

        # 右边界错误：最后一个单词不同
        if pred_words[-1] != true_words[-1]:
            right_errors.append((pred_entity, true_entity))

    # 统计漏检实体（真实实体中未被匹配的）
    miss_entities = [true_list[t_idx] for t_idx in range(true_len) if t_idx not in matched_true]

    # 统计预测无重叠实体（未匹配的预测实体，且与所有真实实体无重叠）
    pred_no_overlap = []
    for p_idx in range(pred_len):
        if p_idx in matched_pred:
            continue  # 已匹配的实体有重叠，排除
        pred_words = pred_entities[p_idx]
        has_overlap = False
        for true_words in true_entities:
            if overlap_count(pred_words, true_words) > 0:
                has_overlap = True
                break
        if not has_overlap:
            pred_no_overlap.append(pred_list[p_idx])

    # 整理结果
    result = {
        "left_error_count": len(left_errors),
        "right_error_count": len(right_errors),
        "miss_count": len(miss_entities),
        "pred_no_overlap_count": len(pred_no_overlap),
        "details": {
            "left_errors": left_errors,
            "right_errors": right_errors,
            "miss_entities": miss_entities,
            "pred_no_overlap_entities": pred_no_overlap
        }
    }
    return result


def NEREval(pred_list, true_list):
    
    analysis = ner_error_analysis(pred_list, true_list)
   
    head_wrong, inside_wrong, wrong_class, miss = analysis['left_error_count'], analysis['right_error_count'], analysis['pred_no_overlap_count'], analysis['miss_count']

    return head_wrong, inside_wrong, wrong_class, miss

