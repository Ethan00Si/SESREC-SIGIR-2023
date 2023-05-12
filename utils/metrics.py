from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd



def ndcg_score(y_true, y_score, k=10):
    """Computing ndcg score metric at k.

    Args:
        y_true (np.ndarray): ground-truth labels.
        y_score (np.ndarray): predicted labels.

    Returns:
        np.ndarray: ndcg scores.
    """
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def dcg_score(y_true, y_score, k=10):
    """Computing dcg score metric at k.

    Args:
        y_true (np.ndarray): ground-truth labels.
        y_score (np.ndarray): predicted labels.

    Returns:
        np.ndarray: dcg scores.
    """
    k = min(np.shape(y_true)[-1], k)
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def hit_score(y_true, y_score, k=10):
    """Computing hit score metric at k.

    Args:
        y_true (np.ndarray): ground-truth labels.
        y_score (np.ndarray): predicted labels.

    Returns:
        np.ndarray: hit score.
    """
    ground_truth = np.where(y_true == 1)[0]
    argsort = np.argsort(y_score)[::-1][:k]
    for idx in argsort:
        if idx in ground_truth:
            return 1
    return 0

def mrr_score(y_true, y_score):
    """Computing mrr score metric.

    Args:
        y_true (np.ndarray): ground-truth labels.
        y_score (np.ndarray): predicted labels.
    
    Returns:
        np.ndarray: mrr scores.
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)
        
def cal_wauc(df, weight):
    
    weight["auc"] = df.groupby("users").apply(groupby_auc)
    wauc_score = (weight["weight"]*weight["auc"]).sum()
    weight.drop(columns="auc", inplace=True)

    return wauc_score


def groupby_auc(df):

    y_hat = df.preds
    y = df.labels
    return roc_auc_score(y, y_hat)


class judger(object):
    def __init__(self):
        super().__init__()

    def cal_metric(self, preds, labels, metrics):
        '''
        codes of evaluation metrics referred from https://github.com/microsoft/recommenders/blob/b704c420ee20b67a9d756ddbfdf5c9afd04b576b/recommenders/models/deeprec/deeprec_utils.py#L514
        '''

        res = dict()
        
        for metric in metrics:
            if metric == "auc":
                auc = roc_auc_score(np.asarray(labels), np.asarray(preds))
                res["auc"] = round(auc, 5)
            elif metric.startswith("ndcg"):  # format like:  ndcg@2;4;6;8
                ndcg_list = [1, 2]
                ks = metric.split("@")
                if len(ks) > 1:
                    ndcg_list = [int(token) for token in ks[1].split(";")]
                for k in ndcg_list:
                    ndcg_temp = np.mean(
                        [
                            ndcg_score(each_labels, each_preds, k)
                            for each_labels, each_preds in zip(labels, preds)
                        ]
                    )
                    res["ndcg@{0}".format(k)] = round(ndcg_temp, 5)
            elif metric.startswith("hit"):  # format like:  hit@2;4;6;8
                hit_list = [1, 2]
                ks = metric.split("@")
                if len(ks) > 1:
                    hit_list = [int(token) for token in ks[1].split(";")]
                for k in hit_list:
                    hit_temp = np.mean(
                        [
                            hit_score(each_labels, each_preds, k)
                            for each_labels, each_preds in zip(labels, preds)
                        ]
                    )
                    res["hit@{0}".format(k)] = round(hit_temp, 5)
            elif metric == "mrr":
                mean_mrr = np.mean(
                    [
                        mrr_score(each_labels, each_preds)
                        for each_labels, each_preds in zip(labels, preds)
                    ]
                )
                res["mrr"] = round(mean_mrr, 5)

        return res

    def cal_weighted_metric(self, users, preds, labels, metrics):
    
        res = {}
        if not metrics:
            return res

        df = pd.DataFrame({'users': users, 'preds': preds, 'labels': labels})
        weight = df[["users", "labels"]].groupby("users").count().reset_index().set_index("users", drop=True).rename(columns={"labels": "weight"})
        weight["weight"] = weight["weight"]/weight["weight"].sum()
        for metric in metrics:
            if metric == 'wauc':
                wauc = cal_wauc(df, weight)
                res["wauc"] = round(wauc, 5)
            else:
                raise ValueError("not define this metric {0}".format(metric))
        return res
