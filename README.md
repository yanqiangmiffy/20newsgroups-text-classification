# 20newsgroups-text-classification
对20 newsgroups 数据集 进行文本分类
# 方法
- 基于传统机器学习方法的文本分类
- 基于深度学习的文本分类
# 测试结果
- 传统机器学习方法

    MultinomialNB准确率为： 0.8960196779964222

    SGDClassifier准确率为： 0.9724955277280859

    LogisticRegression准确率为： 0.9304561717352415

    SVC准确率为： 0.13372093023255813

    LinearSVC准确率为： 0.9749552772808586

    LinearSVR准确率为： 0.00022361359570661896

    MLPClassifier准确率为： 0.9758497316636852

    KNeighborsClassifier准确率为： 0.45840787119856885

    RandomForestClassifier准确率为： 0.9680232558139535

    GradientBoostingClassifier准确率为： 0.9186046511627907

    AdaBoostClassifier准确率为： 0.5916815742397138

    DecisionTreeClassifier准确率为： 0.9758497316636852

- CNN实现文本分类

    需要词向量http://nlp.stanford.edu/data/glove.6B.zip

    效果其实不好...
