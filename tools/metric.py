import numpy as np


class _StreamMetrics(object):
    def __init__(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def update(self, gt, pred):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def get_results(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def to_str(self, metrics):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def reset(self):
        """ Overridden by subclasses """
        raise NotImplementedError()


class StreamSegMetrics(_StreamMetrics):
    """
    Stream Metrics for Image Classfication Task
    """

    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten())

    @staticmethod
    def to_str(results):
        string = "\n"
        for k, v in results.items():
            if k != "Class IoU":
                string += "%s: %f\n" % (k, v)

        return string

    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def get_results(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix

        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)

        precision = np.diag(hist).sum() /  hist.sum(axis=0).sum()
        precision_cls = np.diag(hist) /  hist.sum(axis=0)

        Recall = np.diag(hist).sum() / hist.sum(axis=1).sum()
        Recall_cls = np.diag(hist) / hist.sum(axis=1)

        F1 = (2 * precision * Recall) / (precision + Recall)
        F1_cls = (2 * precision_cls * Recall_cls) / (precision_cls + Recall_cls)

        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

        return {
            "Acc": acc,
            "Acc Class": acc_cls,
            "Precision": precision,
            "Precision Class": precision_cls,
            "Recall": Recall,
            "Recall Class": Recall_cls,
            "F1": F1,
            "F1 Class": F1_cls,
            "FreqW Acc": fwavacc,
        }

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))