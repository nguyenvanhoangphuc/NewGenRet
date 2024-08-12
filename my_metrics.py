from collections import defaultdict


def balance(code, prefix=None, ncentroids=10):
    if prefix is not None:
        prefix = [str(x) for x in prefix]
        prefix_code = defaultdict(list)
        for c, p in zip(code, prefix):
            prefix_code[p].append(c)
        scores = []
        for p, p_code in prefix_code.items():
            scores.append(balance(p_code, ncentroids=ncentroids))
        return {'Avg': sum(scores) / len(scores), 'Max': max(scores), 'Min': min(scores), 'Flat': balance(code)}
    num = [code.count(i) for i in range(ncentroids)]
    base = len(code) // ncentroids
    move_score = sum([abs(j - base) for j in num])
    score = 1 - move_score / len(code) / 2
    return score


def conflict(code, prefix=None):
    if prefix is not None:
        prefix = [str(x) for x in prefix]
        code = [f'{p}{c}' for c, p in zip(code, prefix)]
    code = [str(c) for c in code]
    freq_count = defaultdict(int)
    for c in code:
        freq_count[c] += 1
    max_value = max(list(freq_count.values()))
    min_value = min(list(freq_count.values()))
    len_set = len(set(code))
    return {'Max': max_value, 'Min': min_value, 'Type': len_set, '%': len_set / len(code)}


def ress(code, prefix=None):
    if prefix is not None:
        prefix = [str(x) for x in prefix]
        code = [f'{p}{c}' for c, p in zip(code, prefix)]
    freq_count = defaultdict(int)
    for c in code:
        freq_count[c] += 1
    freq_count = [y for x, y in freq_count.items()]
    freq_count.sort()
    return freq_count


def ress_by_prefix(code, prefix=None):
    freq_count = defaultdict(list)
    for c, p in zip(code, prefix):
        p = str(p)
        freq_count[p].append(c)
    freq_count = [[len(v), len(set(v))] for k, v in freq_count.items()]
    freq_count.sort(key=lambda x: x[1])
    return freq_count

def eval_recall(predictions, labels, subset=None):
    from eval import eval_all
    if subset is not None:
        predictions = [predictions[j] for j in subset]
        labels = [labels[j] for j in subset]
    labels = [[x] for x in labels]
    return eval_all(predictions, labels)