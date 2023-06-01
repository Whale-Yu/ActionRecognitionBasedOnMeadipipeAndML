import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

# KNN、逻辑回归、决策树、随即森林算法
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def load_data():
    # 读取数据
    oath = pd.read_csv("data/oath.csv")
    put_hand = pd.read_csv("data/put_hand.csv")
    stand = pd.read_csv("data/stand.csv")

    # 合并数据-上下堆叠
    res = pd.concat([oath, put_hand, stand])
    print(f'------  数据集  ------\nall_data:{len(res)}')

    # 取数据和目标值
    data = res.iloc[:, 1:]
    target = res.iloc[:, 0]
    # print(data, target)
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(data.values, target.values, test_size=0.3, random_state=0)
    print(f'X_train:{len(X_train)} y_train:{len(y_train)} , X_test:{len(X_test)} y_test:{len(y_test)}\n')

    return X_train, X_test, y_train, y_test


def train_model(model, save_model_file='./models/default.joblib'):
    # 数据集加载
    X_train, X_test, y_train, y_test = load_data()

    # 实例化模型
    pose = model
    # 调用fit方法 训练模型
    pose.fit(X_train, y_train)
    # 保存模型
    joblib.dump(pose, filename=save_model_file)

    # 模型评估
    print('------  模型评估  ------')
    # 方法1：直接比对真实值和预测值
    y_predict = pose.predict(X_test)
    y_predict_sc = pose.predict_proba(X_test)
    print("y_test:\n", y_test)
    print("y_predict:\n", y_predict)
    print("score:\n", y_predict_sc)
    print("比对真实值和预测值:\n", y_test == y_predict)

    # 方法2：计算准确率
    score = pose.score(X_test, y_test)
    print("准确率为：\n", score)


if __name__ == '__main__':
    # 选择KNN、逻辑回归、决策树、随机森林等算法
    # model = KNeighborsClassifier(n_neighbors=3, algorithm='auto')
    # model = LogisticRegression()
    model = DecisionTreeClassifier(criterion="entropy")
    # model=RandomForestClassifier()

    # 保存模型文件名
    save_model_file = 'models/DecisionTreeClassifier.joblib'

    # 训练
    train_model(model)
