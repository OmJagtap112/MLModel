from sklearn import tree

# Machine Learning Prediction ALgorithm

'''
java= 0 for no knowlwdge;1 for core; 2 for advance;
python= 0 for no knowlwdge;1 for core; 2 for advance;
db= 0 for no knowlwdge;1 for core; 2 for advance;
cloud= 0 for no knowledge; 1 for having knowledge;
engi background= 1 for cs/it & 0 for other;
'''
x=[[0,1,0,0,0],[1,0,0,0,0],[0,0,0,0,1],[0,1,1,0,1],[1,1,1,0,0],[1,1,1,0,1],[1,1,0,0,1],[1,1,2,1,1],[0,2,1,1,1],[2,1,0,0,1],[2,0,1,0,1],[2,0,1,1,1],[0,2,2,1,0],[2,2,2,1,1],[1,2,2,1,1],[0,2,1,1,0],[1,1,2,1,1]]
jobs=['Basic Job','Basic Job','Basic Job','Testing','Testing','Testing','Testing','Analysis','Analysis','Development','Development','Development','Data Engineer','Data Enginner','Data Engineer','Data Management','Data Management']

clf=tree.DecisionTreeClassifier()
clf.fit(x,jobs)

#java(0,1,2);python(0,1,2);db(0,1,2);cloud(0,1);background(0,1)
print(clf.predict([[2,0,1,1,1]]))
print(clf.predict([[0,2,2,1,1]]))
print(clf.predict([[1,1,0,0,1]]))
print(clf.predict([[2,2,1,1,1]]))
print(clf.predict([[0,2,2,1,0]]))
print(clf.predict([[1,1,2,1,0]]))
print(clf.predict([[0,1,1,1,0]]))
