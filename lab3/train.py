from feature import NPDFeature
from ensemble import AdaBoostClassifier
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import numpy as np

if __name__ == "__main__":
    def Feature_extract():
        #提取特征
        face_path = '.\\datasets\\original\\face\\face_%03d.jpg'
        faces_path = []
        for i in range(500):
            faces_path.append(face_path % i)

        nonface_path = '.\\datasets\\original\\nonface\\nonface_%03d.jpg'
        nonfaces_path = []
        for i in range(500):
            nonfaces_path.append(nonface_path % i)

        train = np.zeros((1000,165600))
        for i in range(500):
            img = Image.open(faces_path[i])
            img = img.convert('L').resize((24, 24))
            nf = NPDFeature(np.array(img))
            train[i*2] = nf.extract()

            img = Image.open(nonfaces_path[i])
            img = img.convert('L').resize((24, 24))
            nf = NPDFeature(np.array(img))
            train[i*2+1] = nf.extract()
        AdaBoostClassifier.save(train,'train.txt')

    try:
        X = AdaBoostClassifier.load("train.txt")
    except IOError:
        Feature_extract()
        X = AdaBoostClassifier.load("train.txt")

    Y = np.zeros((1000,1))
    for i in range(1000):
        Y[i] = (i+1) % 2
    Y = np.where(Y>0,1,-1)

    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)

    booster = AdaBoostClassifier(DecisionTreeClassifier,15)
    booster.fit(X_train,Y_train)
    predict = booster.predict(X_test)
    wrong_count = 0
    for j in range(predict.shape[0]):
        if predict[j] != Y_test[j]:
            wrong_count += 1
    AdaBoostClassifier.save(classification_report(Y_test, predict),"classifier_report.txt")
    pass

