# encoding: utf-8

import time
import torch
import torch.nn            as nn
import torch.nn.functional as F
import numpy               as np

from sklearn      import metrics
from utils        import get_time_dif
from tensorboardX import SummaryWriter


# TODO:
# init weight
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass



def accuracy(y, y_predict):
    y_true    = y.data.cpu()
    y_predict = torch.max(y_predict.data, 1)[1].cpu()
    return metrics.accuracy_score(y_true, y_predict)



# TODO: rewrite by accuracy()
def evaluate(config, model, dataset, verbose=False):
    # TODO:
    model.eval()

    loss       = 0
    y_predicts = np.array([], dtype=int)
    ys         = np.array([], dtype=int)

    with torch.no_grad():
        for X, y in dataset:
            # predict
            y_predict = model(X)

            # loss
            loss += F.cross_entropy(y_predict, y)

            # concate
            y          = y.data.cpu().numpy()
            y_predict  = torch.max(y_predict.data, 1)[1].cpu().numpy()
            ys         = np.append(ys, y)
            y_predicts = np.append(y_predicts, y_predict)


    # evaluate
    acc      = metrics.accuracy_score(ys, y_predicts)
    loss_avg = loss / len(dataset)

    if verbose:
        classify_report = metrics.classification_report(ys, y_predicts, target_names=config.class_list, digits=4)
        confusion       = metrics.confusion_matrix(ys, y_predicts)
        return acc, loss_avg, classify_report, confusion

    return acc, loss_avg




def test(config, model, test_iter):

    model.load_state_dict(torch.load(config.path_model))
    model.eval()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, verbose=True)

    msg = 'test loss: {0:>5.2},  test acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("precision, recall and f1-score...")
    print(test_report)
    print("confusion matrix...")
    print(test_confusion)




def evaluate_batch(batch_i, y, y_predict, config, model, valid_iter, loss_valid_best, writer):

    if batch_i % 100 == 0:
        acc_train = accuracy(y, y_predict)
        acc_valid, loss_valid = evaluate(config, model, valid_iter)

        if loss_valid < loss_valid_best:
            loss_valid_best = loss_valid
            torch.save(model.state_dict(), config.path_model)
            improve = '*'
            improve_batch = batch_i
        else:
            improve = ''

        # TODO:
        # 0:>6.2%
        # loss.item()
        msg = 'iter: {0:>6},  train loss: {1:>5.2},  train acc: {2:>6.2%},  val loss: {3:>5.2},  val acc: {4:>6.2%},  time: {5} {6}'
        print(msg.format(batch_i, loss.item(), acc_train, loss_valid, acc_valid, time_dif, improve))

        writer.add_scalar("loss/train", loss.item(), batch_i)
        writer.add_scalar("loss/dev"  , loss_valid , batch_i)
        writer.add_scalar("acc/train" , acc_train  , batch_i)
        writer.add_scalar("acc/dev"   , acc_valid  , batch_i)

        # TODO:
        model.train()

        return loss_valid_best, improve, improve_batch



def nomore_improve(batch_i, improve_batch, config):
    if batch_i - improve_batch > config.batch_num_reqimp:
        print('no more improve under %s batch' % config.batch_num_reqimp)
        return True
    return False



def message_epoch(epoch, config):
    print('epoch [{}/{}]'.format(epoch + 1, config.epoches_num))



def train(config, model, train_iter, valid_iter, test_iter):
    """
    Parameters
    ----------
    X: (words_idx, num_valid_words, bigram, trigram)
    """

    # TODO:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # decay learning rate each epoch: lr = gamma * lr
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    batch_i         = 0
    loss_valid_best = float('inf')
    improve_batch   = 0
    path_log        = config.path_log + '/' + time.strftime('%m-%d_%H.%M', time.localtime()

    with SummaryWriter(log_dir=path_log) as writer:
        for epoch in range(config.epoches_num):
            message_epoch()
            # scheduler.step() # decay learning rate

            for i, (X, y) in enumerate(train_iter):
                y_predict = model(X)
                model.zero_grad() # TODO:
                loss = F.cross_entropy(y_predict, y)
                loss.backward()
                optimizer.step()

                loss_valid_best, improve, improve_batch = evaluate_batch(\
                    batch_i, y, y_predict, config, model, valid_iter, loss_valid_best, writer)

                batch_i += 1

                if nomore_improve(batch_i, improve_batch, config): break
            if nomore_improve(batch_i, improve_batch, config): break

    test(config, model, test_iter)
