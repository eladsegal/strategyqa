import numpy as np

MAX_STEPS = 10


def _index_to_reference(i):
    return f"#{i + 1}"


def fill_in_references(decomposition_step, step_answers):
    for i in range(MAX_STEPS - 1, -1, -1):
        ref = _index_to_reference(i)
        if i < len(step_answers) and step_answers[i] is not None:
            decomposition_step = decomposition_step.replace(ref, step_answers[i])
    return decomposition_step


def has_reference(decomposition_step):
    for i in range(MAX_STEPS - 1, -1, -1):
        ref = _index_to_reference(i)
        if ref in decomposition_step:
            return True
    return False


def get_references(decomposition_step):
    refs = []
    for i in range(MAX_STEPS - 1, -1, -1):
        ref = _index_to_reference(i)
        if ref in decomposition_step:
            refs.append(i)
    return refs


def get_reachability(decomposition):
    ref_graph = np.zeros((len(decomposition), len(decomposition)))
    for i, step in enumerate(decomposition):
        refs = get_references(step)
        if i in refs:
            return None
        ref_graph[i][refs] = 1

    length = 1
    reachability = ref_graph
    while True:
        length += 1
        step_reachability = np.linalg.matrix_power(reachability, length)
        if np.sum(step_reachability) == 0:
            break
        if length == MAX_STEPS:
            return None
        reachability += step_reachability
    return reachability
