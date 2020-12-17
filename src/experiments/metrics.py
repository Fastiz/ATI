

def best_n_metric(n):
    def metric(matches):
        return sum([m.distance for m in matches[0:n]])
    return metric


def threshold_metric(threshold):
    def metric(matches):
        return sum(map(lambda m: m.distance < threshold, matches))
    return metric


def distance_sum_metric():
    def metric(matches):
        return best_n_metric(len(matches))(matches)
    return metric


def distance_avg_metric():
    def metric(matches):
        return distance_sum_metric()(matches) / float(len(matches))
    return metric
