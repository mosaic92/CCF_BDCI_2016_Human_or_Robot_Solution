# CCF_BDCI_2016_Human_or_Robot_Solution
###  代码说明

#### 代码文件

* feature_train.py：训练集特征工程的代码文件
* feature_test.py：测试集特征工程的代码文件
* modeling.py：训练模型、模型融合和预测结果的代码文件

#### 代码目录

* 代码文件中的    
  *path = '…/AdMaster_competition_dataset/admaster/'*   
  为赛题提供数据文件所在目录
* path目录下的train3文件夹为训练集特征工程代码的结果输出路径
* path目录下的test文件夹为测试集特征工程代码的结果输出路径
* 训练所得模型和预测结果均保存在path目录下

#### 其他注意事项

LightGBM为微软2016年10月11日开源的机器学习算法，其在2016年12月02日提供了python接口beta版本（版本号为0.1），近期一直在继续更新，变动较大，若调用该python库的函数报错，请参考https://github.com/Microsoft/LightGBM/blob/master/docs/Python-API.md，进行相应的修改。
​
