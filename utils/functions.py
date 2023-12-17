import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from torch import optim

def dict_to_str(src_dict):
    dst_str = ""
    for key in src_dict.keys():
        dst_str += " %s: %.4f " %(key, src_dict[key])
    return dst_str

def multiclass_acc(y_pred, y_true):
    """
    Compute the multiclass accuracy w.r.t. groundtruth

    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    return np.sum(np.round(y_pred) == np.round(y_true)) / float(len(y_true))

def eval_regression(y_pred, y_true, exclude_zero=False):
    test_preds = y_pred.view(-1).cpu().detach().numpy()
    test_truth = y_true.view(-1).cpu().detach().numpy()

    test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
    test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
    test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
    test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)
    test_preds_a3 = np.clip(test_preds, a_min=-1., a_max=1.)
    test_truth_a3 = np.clip(test_truth, a_min=-1., a_max=1.)

    mae = np.mean(np.absolute(test_preds - test_truth))  # Average L1 distance between preds and truths
    corr = np.corrcoef(test_preds, test_truth)[0][1]
    mult_a7 = multiclass_acc(test_preds_a7, test_truth_a7)
    mult_a5 = multiclass_acc(test_preds_a5, test_truth_a5)
    mult_a3 = multiclass_acc(test_preds_a3, test_truth_a3)

    non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])
    non_zeros_binary_truth = (test_truth[non_zeros] > 0)
    non_zeros_binary_preds = (test_preds[non_zeros] > 0)

    non_zeros_acc2 = accuracy_score(non_zeros_binary_preds, non_zeros_binary_truth)
    non_zeros_f1_score = f1_score(non_zeros_binary_truth, non_zeros_binary_preds, average='weighted')

    binary_truth = (test_truth >= 0)
    binary_preds = (test_preds >= 0)
    acc2 = accuracy_score(binary_preds, binary_truth)
    f_score = f1_score(binary_truth, binary_preds, average='weighted')

    eval_results = {
        "Has0_acc_2": round(acc2, 4),
        "Has0_F1_score": round(f_score, 4),
        "Non0_acc_2": round(non_zeros_acc2, 4),
        "Non0_F1_score": round(non_zeros_f1_score, 4),
        "Mult_acc_5": round(mult_a5, 4),
        "Mult_acc_7": round(mult_a7, 4),
        "MAE": round(mae, 4),
        "Corr": round(corr, 4)
    }
    return eval_results

class Storage(dict):
    """
    A Storage object is like a dictionary except `obj.foo` can be used inadition to `obj['foo']`
    ref: https://blog.csdn.net/a200822146085/article/details/88430450
    """
    def __getattr__(self, key):
        try:
            return self[key] if key in self else False
        except KeyError as k:
            raise AttributeError(k)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __str__(self):
        return "<" + self.__class__.__name__ + dict.__repr__(self) + ">"

def prep_optimizer(args, model):

    bert_no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    bert_params = list(model.text_model.named_parameters())
    audio_params = list(model.audio_model.named_parameters())
    video_params = list(model.video_model.named_parameters())

    bert_params_decay = [p for n, p in bert_params if not any(nd in n for nd in bert_no_decay)]
    bert_params_no_decay = [p for n, p in bert_params if any(nd in n for nd in bert_no_decay)]
    audio_params = [p for n, p in audio_params]
    video_params = [p for n, p in video_params]

    model_params_other = [p for n, p in list(model.named_parameters()) if 'text_model' not in n and \
                            'audio_model' not in n and 'video_model' not in n and 'TA_MI_net' not in n and \
                            'TV_MI_net' not in n and 'VA_MI_net' not in n]

    optimizer_grouped_parameters = [
        {'params': bert_params_decay, 'weight_decay': args.weight_decay_bert,
         'lr': args.learning_rate_bert},
        {'params': bert_params_no_decay, 'weight_decay': 0.0, 'lr': args.learning_rate_bert},
        {'params': audio_params, 'weight_decay': args.weight_decay_audio, 'lr': args.learning_rate_audio},
        {'params': video_params, 'weight_decay': args.weight_decay_video, 'lr': args.learning_rate_video},
        {'params': model_params_other, 'weight_decay': args.weight_decay_other,
         'lr': args.learning_rate_other}
    ]
    optimizer = optim.Adam(optimizer_grouped_parameters)

    scheduler = None
    return optimizer, scheduler, model
