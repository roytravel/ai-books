from sklearn.svm import SVC

# scikit-learn에선 SVM 모델의 오류 허용 범위를 파라미터 C로 지정가능 (Default = 1)
# 값이 클 수록 Hard Margin(오류 허용 x), 작을수록 Soft Margin(오류 허용)
# kernel 파라미터를 'poly'로 표현하여 초평면의 Decision Boundary를 얻을 수 있음(2차원 -> 3차원 변환)
classifier = SVC(kernel = 'linear', C = 0.01)

training_points = [[1, 2], [1, 5], [2, 2], [7, 5], [9, 4], [8, 2]]
labels = [1, 1, 1, 0, 0, 0]

classifier.fit(training_points, labels)

print (classifier.predict([[3,2]]))

# Deicison Boundary를 결정하는 Support Vector 확인
print (classifier.support_vectors_)
