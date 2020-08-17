import os
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear
import numpy as np
import Data_Load_Module
import Network


#调用加载数据的函数
train_loader = Data_Load_Module.load_data('train')

    
#在使用GPU机器时，可以将use_gpu变量设置成True
use_gpu = True
place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()

with fluid.dygraph.guard(place):
    model = Network.MNIST()
    model.train() 
    
    #四种优化算法的设置方案，可以逐一尝试效果
    #optimizer = fluid.optimizer.SGDOptimizer(learning_rate=0.01, parameter_list=model.parameters())
    #optimizer = fluid.optimizer.MomentumOptimizer(learning_rate=0.01, momentum=0.9, parameter_list=model.parameters())
    #optimizer = fluid.optimizer.AdagradOptimizer(learning_rate=0.01, parameter_list=model.parameters())
    optimizer = fluid.optimizer.AdamOptimizer(learning_rate=0.01, parameter_list=model.parameters())
    
    EPOCH_NUM = 5
    for epoch_id in range(EPOCH_NUM):
        for batch_id, data in enumerate(train_loader()):
            #准备数据
            image_data, label_data = data
            image = fluid.dygraph.to_variable(image_data)
            label = fluid.dygraph.to_variable(label_data)
            
            #前向计算的过程，同时拿到模型输出值和分类准确率
            predict, acc = model(image, label)
            
            #计算损失，取一个批次样本损失的平均值
            loss = fluid.layers.cross_entropy(predict, label)
            avg_loss = fluid.layers.mean(loss)
            
            #每训练了200批次的数据，打印下当前Loss的情况
            if batch_id % 200 == 0:
                print("epoch: {}, batch: {}, loss is: {}, acc is {}".format(epoch_id, batch_id, avg_loss.numpy(), acc.numpy()))
            
            #后向传播，更新参数的过程
            avg_loss.backward()
            optimizer.minimize(avg_loss)
            model.clear_gradients()

    #保存模型参数
    fluid.save_dygraph(model.state_dict(), 'mnist')



with fluid.dygraph.guard():
    print('start evaluation .......')
    #加载模型参数
    model = Network.MNIST()
    model_state_dict, _ = fluid.load_dygraph('mnist')
    model.load_dict(model_state_dict)

    model.eval()
    eval_loader = Data_Load_Module.load_data('eval')

    acc_set = []
    avg_loss_set = []
    for batch_id, data in enumerate(eval_loader()):
        x_data, y_data = data
        img = fluid.dygraph.to_variable(x_data)
        label = fluid.dygraph.to_variable(y_data)
        prediction, acc = model(img, label)
        loss = fluid.layers.cross_entropy(input=prediction, label=label)
        avg_loss = fluid.layers.mean(loss)
        acc_set.append(float(acc.numpy()))
        avg_loss_set.append(float(avg_loss.numpy()))
    
    #计算多个batch的平均损失和准确率
    acc_val_mean = np.array(acc_set).mean()
    avg_loss_val_mean = np.array(avg_loss_set).mean()

    print('loss={}, acc={}'.format(avg_loss_val_mean, acc_val_mean))