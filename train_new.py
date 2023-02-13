import os
import numpy as np
import config
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import json
from parser_tool import parameter_parser
from torch.autograd import Variable
from models_update import move_data_to_gpu, BytecodeNet, SBFusionNet
from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print('using torch', torch.__version__)
args = parameter_parser()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# Loss function
# adversarial_loss = torch.nn.BCELoss()

print(
    args.dataset, args.m, args.lr, args.cuda, args.epoch, args.seed
)

# start here
def main():
    # Binary classification
    classes_num = len(config.lbs)

    # Paths
    workspace = './'
    train_sourcecode_path = os.path.join(workspace, 'sourcecode', args.dataset, 'training.json')
    train_bytecode_path = os.path.join(workspace, 'bytecode', args.dataset, args.m + '_binary_train.json')
    eval_sourcecode_path = os.path.join(workspace, 'sourcecode', args.dataset, 'testing.json')
    eval_bytecode_path = os.path.join(workspace, 'bytecode', args.dataset, args.m + '_binary_test.json')

    # load models
    student = BytecodeNet(classes_num)
    teacher = SBFusionNet(classes_num)

    # load optimizers
    student_optimizer = optim.Adam(student.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.wd)
    teacher_optimizer = optim.Adam(teacher.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.wd)

    #######################################  load train data  #######################################

    # load source code and bytecode of training set respectively
    load_sourcecode = open(train_sourcecode_path, 'r')
    train_sourcecode_dict = json.load(load_sourcecode)
    # train_sourcecode_names = np.array([item['contract_name'] for item in sourcecode_dict])
    train_sourcecode_x = np.array([item['node_features'] for item in train_sourcecode_dict])
    train_sourcecode_y = np.array([item['targets'] for item in train_sourcecode_dict])
    load_sourcecode.close()

    load_bytecode = open(train_bytecode_path, 'r')
    train_bytecode_dict = json.load(load_bytecode)
    # train_bytecode_names = np.array([item['contract_name'] for item in bytecode_dict])
    train_bytecode_x = np.array([item['node_features'] for item in train_bytecode_dict])
    # train_bytecode_y = np.array([item['targets'] for item in bytecode_dict])
    load_bytecode.close()

    # Data format conversion
    train_sourcecode_x = torch.tensor(train_sourcecode_x).float()
    train_bytecode_x = torch.tensor(train_bytecode_x).float()
    train_sourcecode_y = train_sourcecode_y.tolist()
    train_sourcecode_y = [int(num) for num in train_sourcecode_y]
    train_sourcecode_y = torch.tensor(train_sourcecode_y).float()

    # training data loader
    torch_dataset = Data.TensorDataset(train_sourcecode_x, train_bytecode_x, train_sourcecode_y)
    train_loader = Data.DataLoader(dataset=torch_dataset, batch_size=args.batch_size, shuffle=args.shuffle,
                                   num_workers=2)

    #######################################  load test data  #######################################

    # load source code and bytecode of testing set respectively
    load_sourcecode = open(eval_sourcecode_path, 'r')
    eval_sourcecode_dict = json.load(load_sourcecode)
    # eval_sourcecode_names = np.array([item['contract_name'] for item in sourcecode_dict])
    eval_sourcecode_x = np.array([item['node_features'] for item in eval_sourcecode_dict])
    eval_sourcecode_y = np.array([item['targets'] for item in eval_sourcecode_dict])
    load_sourcecode.close()

    load_bytecode = open(eval_bytecode_path, 'r')
    eval_bytecode_dict = json.load(load_bytecode)
    # eval_bytecode_names = np.array([item['contract_name'] for item in bytecode_dict])
    eval_bytecode_x = np.array([item['node_features'] for item in eval_bytecode_dict])
    # eval_bytecode_y = np.array([item['targets'] for item in bytecode_dict])
    load_bytecode.close()

    # Data format conversion
    eval_sourcecode_x = torch.tensor(eval_sourcecode_x).float()
    eval_bytecode_x = torch.tensor(eval_bytecode_x).float()
    eval_sourcecode_y = eval_sourcecode_y.tolist()
    eval_sourcecode_y = [int(num) for num in eval_sourcecode_y]
    eval_sourcecode_y = torch.tensor(eval_sourcecode_y).float()

    # training data loader
    torch_dataset = Data.TensorDataset(eval_sourcecode_x, eval_bytecode_x, eval_sourcecode_y)
    test_loader = Data.DataLoader(dataset=torch_dataset, batch_size=args.batch_size, shuffle=args.shuffle,
                                  num_workers=2)

    cuda = args.cuda
    if cuda:
        student.cuda()
        teacher.cuda()

    # training
    def train(train_loader):
        teacher_loss, student_loss = 0, 0
        Sgtl = student.s_gtl
        Smll = student.s_mll
        Sbbl = student.s_bbl
        Sbsl = student.s_bsl

        Tgtl = teacher.t_gtl
        Tmll = teacher.t_mll
        Tbbl = teacher.t_bbl
        Tbsl = teacher.t_bsl

        for (iteration, (train_batch_sourcecode_x, train_batch_bytecode_x, train_batch_y)) in enumerate(train_loader):
            train_batch_sourcecode_x = move_data_to_gpu(train_batch_sourcecode_x, cuda)
            train_batch_bytecode_x = move_data_to_gpu(train_batch_bytecode_x, cuda)
            train_batch_y = move_data_to_gpu(train_batch_y, cuda)

            student.train()
            teacher.train()

            # teacher 1
            teacher_predict, teacher_predict_b, teacher_predict_s = teacher(train_batch_sourcecode_x,
                                                                            train_batch_bytecode_x)
            # teacher_predict = Variable(teacher_predict.detach().data, requires_grad=False)
            # teacher_predict_b = Variable(teacher_predict_b.detach().data, requires_grad=False)
            # teacher_predict_s = Variable(teacher_predict_s.detach().data, requires_grad=False)

            # student 1
            student_predict, student_predict_inter_rep, student_predict_transformed = student(train_batch_bytecode_x)
            # student_predict = Variable(student_predict.detach().data, requires_grad=False)
            # student_predict_inter_rep = Variable(student_predict_inter_rep.detach().data, requires_grad=False)
            # student_predict_transformed = Variable(student_predict_transformed.detach().data, requires_grad=False)

            teacher.train()
            student.train()

            # teacher 2
            teacher_output, teacher_output_b, teacher_output_s = teacher(train_batch_sourcecode_x,
                                                                         train_batch_bytecode_x)

            teacher_output = torch.argmax(teacher_output, dim=1, keepdim=False).float()
            # student_predict = torch.argmax(student_predict, dim=1, keepdim=False).float()

            train_batch_y = train_batch_y.reshape(-1, 1)
            # print(train_batch_y.shape)

            #
            teacher_loss = \
                Tgtl * F.binary_cross_entropy(teacher_output, train_batch_y) + \
                Tbsl * F.mse_loss(teacher_output_s, student_predict_transformed) + \
                Tbbl * F.mse_loss(teacher_output_b, student_predict_inter_rep)
                # Tmll * F.binary_cross_entropy(teacher_output, student_predict.detach()) + \

            # student 2
            student_output, student_predict_inter_rep, student_output_transformed = student(train_batch_bytecode_x)
            student_output = torch.argmax(student_output, dim=1, keepdim=False).float()
            # teacher_predict = torch.argmax(teacher_predict, dim=1, keepdim=False).float()

            student_loss = \
                Sgtl * F.binary_cross_entropy(student_output, train_batch_y) + \
                Sbsl * F.mse_loss(student_output_transformed, teacher_predict_s) + \
                Sbbl * F.mse_loss(student_predict_inter_rep, teacher_predict_b)
               # Smll * F.binary_cross_entropy(student_output, teacher_predict.detach()) + \

            teacher_optimizer.zero_grad()
            student_optimizer.zero_grad()

            teacher_loss.backward()
            student_loss.backward()
            # sum_loss = teacher_loss + student_loss
            # sum_loss.backward()

            teacher_optimizer.step()
            student_optimizer.step()

        print('teacher loss weights: ground_truth_loss: ({:.6f}), mutual_learning_loss: ({:.6f}), '
              'binary_binary_loss: ({:.6f}), binary_source_loss: ({:.6f})'.format(Tgtl, Tmll, Tbbl, Tbsl)
              )
        print('student loss weights: ground_truth_loss: ({:.6f}), mutual_learning_loss: ({:.6f}), '
              'binary_binary_loss: ({:.6f}), binary_source_loss: ({:.6f})'.format(Sgtl, Smll, Sbbl, Sbsl)
              )
        print('Train Epoch: {}, Teacher Loss: {:.6f}, Student Loss: {:.6f}'.format(
            epoch + 1, teacher_loss.item(), student_loss.item()))

    # testing
    def test(test_loader):
        bytecode_and_sourcecode = 0
        test_loss = 0
        outputs = []
        targets = []

        # Evaluate on mini-batch
        for (step, (eval_batch_sourcecode_x, eval_batch_bytecode_x, eval_batch_y)) in enumerate(test_loader):

            if bytecode_and_sourcecode == 1:
                model = teacher
            else:
                model = student

            eval_batch_sourcecode_x = move_data_to_gpu(eval_batch_sourcecode_x, cuda)
            eval_batch_bytecode_x = move_data_to_gpu(eval_batch_bytecode_x, cuda)
            eval_batch_y = move_data_to_gpu(eval_batch_y, cuda)

            # Predict
            model.eval()

            eval_batch_y = eval_batch_y.reshape(-1, 1)
            # print(eval_batch_y.shape)

            if bytecode_and_sourcecode == 1:
                batch_output, _, _ = model(eval_batch_sourcecode_x, eval_batch_bytecode_x)
                # batch_output = torch.argmax(batch_output, dim=1, keepdim=False).float()
                test_loss = F.binary_cross_entropy(batch_output, eval_batch_y)
                # pred = output.detach().cpu().max(1, keepdim=True)[1]
            else:
                batch_output, _, _ = model(eval_batch_bytecode_x)
                # batch_output = torch.argmax(batch_output, dim=1, keepdim=False).float()
                test_loss = F.binary_cross_entropy(batch_output, eval_batch_y)

            # Append data
            outputs.append(batch_output.data.cpu().numpy())
            targets.append(eval_batch_y.data.cpu().numpy())

        dict = {}
        outputs = np.concatenate(outputs, axis=0)
        dict['outputs'] = outputs
        targets = np.concatenate(targets, axis=0)
        dict['targets'] = targets

        targets = dict['targets']
        outputs = dict['outputs']

        # outputs = (outputs >= 0.5)

        # Data analysis for the predictions
        ap = average_precision_score(targets, outputs, average=None)
        mAP = np.mean(ap)
        auc = roc_auc_score(targets, outputs)
        mAUC = np.mean(auc)

        tn, fp, fn, tp = confusion_matrix(targets, outputs.round()).ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        F1 = (2 * precision * recall) / (precision + recall)
        FPR = fp / (fp + tn)

        print(
            'Test Loss: {:.6f}, Accuracy: ({:.6f}), Recall: ({:.6f}), Precision: ({:.6f}), '
            'F1-Score: ({:.6f}), FPR: ({:.6f}), mAP: ({:.6f}), mAUC: ({:.6f})'.format(
                test_loss.item(), accuracy, recall, precision, F1, FPR, mAP, mAUC)
        )

    print('start training......')
    for epoch in range(args.epoch):
        train(train_loader)

    print('start testing......')
    test(test_loader)


if __name__ == '__main__':
    main()
