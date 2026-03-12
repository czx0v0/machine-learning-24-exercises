import numpy as np
x = np.array([[0,1,3],
             [0,3,1],
             [1,2,2],
             [1,1,3],
             [1,2,3],
             [0,1,2],
             [1,1,2],
             [1,1,1],
             [1,3,1],
             [0,2,1]])
y = np.array([-1,-1,-1,-1,-1,-1,1,1,-1,-1])
dataMatrix = np.matrix(x)
labelMatrix = np.matrix([-1,-1,-1,-1,-1,-1,1,1,-1,-1]).T
def stumpClassify(dataMatrix,dim,threshold,threshInequal):
    # 分类函数，通过阈值threshold对数据进行分类
    returnArray = np.ones((np.shape(dataMatrix)[0],1)) # 初始化
    # 小于：lt less than；大于gt  greater than
    if threshInequal == 'lt':
        returnArray[dataMatrix[:,dim]<=threshold] = -1.0
    else:
        returnArray[dataMatrix[:,dim]>threshold] = -1.0
    return returnArray
def buildStump(dataMatrix,classLabels,D):
    # 构建当前数据集的最佳决策树，遍历不同的阈值且计算分类误差， 找到分类误差最小的单层决策树
    # D:样本权重
    m,n = np.shape(dataMatrix)
    numSteps = 50.0 # 步长
    bestStumpInfo = {} # 最佳单层决策树的信息
    bestClassEst = np.mat(np.zeros((m,1))) # 最佳的分类结果
    minErr = float('inf') # 最小的误差
    for i in range(n):
        # 计算当前第i个特征的最大和最小值
        stepSize = (dataMatrix[:,i].max() - dataMatrix[:,i].min())/numSteps
        for j in range(-1,int(numSteps)+1):
            for inequalType in ['lt','gt']:
                threshold = (dataMatrix[:,i].min() + stepSize*float(j))
                predictVals = stumpClassify(dataMatrix,i,threshold,inequalType)
                # 计算误差，初始化误差矩阵
                errMatrix = np.matrix(np.ones((m,1)))
                errMatrix[classLabels== predictVals] = 0
                weightedErr = np.mean(D.T * errMatrix)
                # print(i,threshhold,inequal,weighted_err)
                print(f'SPLIT[{j}]:dim {i}, threshold {threshold:.2f}, inequalType {inequalType}, weighted error {weightedErr:.3f}')
                if weightedErr < minErr:
                    minErr = weightedErr
                    bestClassEst = predictVals.copy()
                    bestStumpInfo['dim'] = i
                    bestStumpInfo['thresh'] = threshold
                    bestStumpInfo['ineq'] = inequalType
    return bestStumpInfo,minErr,bestClassEst
# 计算权重
D = np.matrix(np.ones((dataMatrix.shape[0],1))/dataMatrix.shape[0])
bestStumpInfo,minErr,bestClassEst = buildStump(dataMatrix,labelMatrix,D)
print(f'bestStumpInfo {bestStumpInfo}')
print(f'minErr {minErr}')
print(f'bestClassEst {bestClassEst.T}')
print(f'labelMatrix {labelMatrix.T}')
