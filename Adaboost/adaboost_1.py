def adaBoostDSTrain(dataMatrix,classLabels,tol = 0.05,iter_max = 40):
    weakClassArray = []
    m = np.shape(dataMatrix)[0]
    D = np.matrix(np.ones((m,1))/m) # 初始化权重
    aggClassEst = np.mat(np.zeros((m,1))) # 初始化估计值
    for i in range(iter_max):
        bestStumpi,erri,classEsti = buildStump(dataMatrix,classLabels,D)
        print(f'D {D.T}',end=',')
        alpha = float(0.5*np.log((1.0-erri)/max(erri,1e-15))) # 避免分母为0
        bestStumpi['alpha'] = alpha # alpha：弱分类器权重
        weakClassArray.append(bestStumpi) #存储弱分类器
        print(f'classEsti {classEsti.T}',end=',')
        expon = np.multiply(-1*alpha*labelMatrix,classEsti) # 指数
        D = np.multiply(D,np.exp(expon))
        D = D/D.sum() # 更新样本权重，除以规范化因子
        aggClassEst += alpha*classEsti
        print(f'aggClassEsti {aggClassEst}',end=',')
        # 计算误差
        aggErr = np.multiply(np.sign(aggClassEst)!=labelMatrix,np.ones((m,1)))
        errRate = aggErr.sum()/m
        print(f'total error {errRate:.3f}')
        # 直到误差为0
        if errRate == 0:
            break
    return weakClassArray,aggClassEst

weakClassArray, aggClassEst = adaBoostDSTrain(dataMatrix, labelMatrix)
print(f'weakClassArray {weakClassArray}')
print(f'aggClassEst {aggClassEst}')
def adaBoostDSClassify(dataMatrix,classifierArray):
    m = dataMatrix.shape[0]
    aggClassEst = np.matrix(np.zeros((m,1)))# 初始化估计值
    # 遍历分类器
    for i in range(len(classifierArray)):
        classEsti = stumpClassify(dataMatrix,classifierArray[i]['dim'],classifierArray[i]['thresh'],classifierArray[i]['ineq'])
        aggClassEst += classifierArray[i]['alpha']*classEsti
        print(f'aggClassEst {aggClassEst.T}')
    return np.sign(aggClassEst)
def cal_error(labelMatrix, predLabelMatrix):
    # 计算错误率
    err = 0
    n = labelMatrix.shape[0]
    for i in range(n):
        if labelMatrix[i] != predLabelMatrix[i]:
            err += 1
    return err / n
aggClassEstAdaDS = adaBoostDSClassify(dataMatrix, weakClassArray)
print(f'aggClassEstAdaDS {aggClassEstAdaDS.T}')
print(f'lableMatrix {labelMatrix.T}')
print(f'error {cal_error(labelMatrix, aggClassEstAdaDS)}')
